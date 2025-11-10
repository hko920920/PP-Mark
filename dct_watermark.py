import argparse
import pathlib
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from reedsolo import RSCodec


Coord = Tuple[int, int]
Pair = Tuple[Coord, Coord]


def _zigzag_coords(size: int = 8) -> List[Coord]:
    coords: List[Coord] = []
    for s in range(0, 2 * size - 1):
        if s % 2 == 0:
            r = min(s, size - 1)
            c = s - r
            while r >= 0 and c < size:
                coords.append((r, c))
                r -= 1
                c += 1
        else:
            c = min(s, size - 1)
            r = s - c
            while c >= 0 and r < size:
                coords.append((r, c))
                r += 1
                c -= 1
    return coords


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bits_to_bytes(bits: Sequence[int]) -> bytes:
    arr = np.array(bits, dtype=np.uint8)
    padded_len = int(np.ceil(len(arr) / 8) * 8)
    if padded_len != len(arr):
        arr = np.pad(arr, (0, padded_len - len(arr)))
    arr = arr.reshape(-1, 8)
    bytes_arr = np.packbits(arr, axis=1)
    return bytes(bytes_arr.flatten().tolist())


@dataclass
class WatermarkConfig:
    target_size: int = 512
    block_size: int = 8
    pairs_per_block: int = 14
    ecc_symbols: int = 96
    delta: float = 60.0
    channel_index: int = 1
    repeat_factor: int = 11
    interleave_seed: int = 1337

    def __post_init__(self) -> None:
        if self.pairs_per_block <= 0:
            raise ValueError("pairs_per_block must be positive")
        if self.ecc_symbols <= 0 or self.ecc_symbols >= 255:
            raise ValueError("ecc_symbols must be in (0,255)")
        if self.delta <= 0:
            raise ValueError("delta must be positive")
        if self.channel_index not in (0, 1, 2):
            raise ValueError("channel_index must be 0 (B), 1 (G), or 2 (R)")
        if self.repeat_factor <= 0:
            raise ValueError("repeat_factor must be positive")
        if self.interleave_seed < 0:
            raise ValueError("interleave_seed must be non-negative")


class DCTWatermarker:
    def __init__(self, config: WatermarkConfig):
        self.config = config
        self.codec = RSCodec(config.ecc_symbols)
        coords = [c for c in _zigzag_coords(config.block_size) if c != (0, 0)]
        if len(coords) < config.pairs_per_block * 2:
            raise ValueError("Not enough DCT coordinates to form the requested number of pairs.")
        self.coeff_pairs: List[Pair] = []
        for i in range(0, config.pairs_per_block * 2, 2):
            self.coeff_pairs.append((coords[i], coords[i + 1]))
        self.blocks_per_dim = self.config.target_size // self.config.block_size

    def _region_block_sets(self) -> List[Tuple[str, List[int], int]]:
        total_blocks = self.blocks_per_dim * self.blocks_per_dim
        full = list(range(total_blocks))
        margin = max(1, self.blocks_per_dim // 10)
        center: List[int] = []
        for row in range(self.blocks_per_dim):
            for col in range(self.blocks_per_dim):
                idx = row * self.blocks_per_dim + col
                if margin <= row < self.blocks_per_dim - margin and margin <= col < self.blocks_per_dim - margin:
                    center.append(idx)
        edge = [idx for idx in full if idx not in center]
        return [
            ("center", center, 1),
            ("edge", edge, 2),
        ]

    def _region_schedule(self, block_indices: List[int], seed_offset: int) -> List[Tuple[int, int]]:
        positions = [
            (block_idx, pair_idx)
            for block_idx in block_indices
            for pair_idx in range(len(self.coeff_pairs))
        ]
        rng = np.random.default_rng(self.config.interleave_seed + seed_offset)
        rng.shuffle(positions)
        return positions

    def embed(self, image_path: pathlib.Path, output_path: pathlib.Path, message: bytes) -> None:
        image = self._load_image(image_path)
        channel = image[:, :, self.config.channel_index].astype(np.float32)
        payload = self._build_payload(message)
        payload_bits = _bytes_to_bits(payload)
        if self.config.repeat_factor > 1:
            payload_bits = np.repeat(payload_bits, self.config.repeat_factor)

        blocks = self._split_blocks(channel)
        dct_blocks = [cv2.dct(block) for block in blocks]

        for _, block_indices, seed_offset in self._region_block_sets():
            capacity = len(block_indices) * len(self.coeff_pairs)
            if len(payload_bits) > capacity:
                raise ValueError("Payload exceeds regional capacity; reduce message length or increase pairs per block.")
            bit_idx = 0
            for block_idx, pair_idx in self._region_schedule(block_indices, seed_offset):
                if bit_idx >= len(payload_bits):
                    break
                a, b = self.coeff_pairs[pair_idx]
                bit = int(payload_bits[bit_idx])
                self._enforce_pair_delta(dct_blocks[block_idx], a, b, bit)
                bit_idx += 1

        spatial_blocks = [cv2.idct(block) for block in dct_blocks]
        channel = self._merge_blocks(spatial_blocks, channel.shape)
        image[:, :, self.config.channel_index] = _ensure_uint8(channel)
        cv2.imwrite(str(output_path), image)

    def extract(self, image_path: pathlib.Path) -> bytes:
        image = self._load_image(image_path)
        channel = image[:, :, self.config.channel_index].astype(np.float32)
        blocks = self._split_blocks(channel)
        dct_blocks = [cv2.dct(block) for block in blocks]
        last_error: Exception | None = None
        for region_name, block_indices, seed_offset in self._region_block_sets():
            try:
                bits = []
                for block_idx, pair_idx in self._region_schedule(block_indices, seed_offset):
                    a, b = self.coeff_pairs[pair_idx]
                    diff = dct_blocks[block_idx][a] - dct_blocks[block_idx][b]
                    bits.append(1 if diff >= 0 else 0)
                return self._decode_from_bits(bits)
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                continue
        raise ValueError(f"All regional decoders failed: {last_error}")

    def _decode_from_bits(self, bits: List[int]) -> bytes:
        raw = np.array(bits, dtype=np.int32)
        repeat = self.config.repeat_factor
        usable = (len(raw) // repeat) * repeat
        if usable == 0:
            raise ValueError("Recovered data shorter than repeat_factor.")
        raw = raw[:usable]
        if repeat > 1:
            grouped = raw.reshape(-1, repeat)
            threshold = (repeat // 2) + 1
            collapsed = (grouped.sum(axis=1) >= threshold).astype(np.uint8)
        else:
            collapsed = raw.astype(np.uint8)

        header_bytes_len = 2 + self.config.ecc_symbols
        header_bit_len = header_bytes_len * 8
        if header_bit_len > len(collapsed):
            raise ValueError("Not enough bits recovered to decode header.")
        header_bits = collapsed[:header_bit_len]
        header_bytes = _bits_to_bytes(header_bits)[:header_bytes_len]
        header_plain = self.codec.decode(header_bytes)[0]
        total_payload_bytes = int.from_bytes(header_plain[:2], "big")
        payload_bit_len = total_payload_bytes * 8
        required_bits = header_bit_len + payload_bit_len
        if required_bits > len(collapsed):
            raise ValueError("Watermark payload truncated; insufficient bits recovered.")
        payload_bits = collapsed[header_bit_len:required_bits]
        payload_bytes = _bits_to_bytes(payload_bits)[:total_payload_bytes]
        decoded = self.codec.decode(payload_bytes)[0]
        msg_len = int.from_bytes(decoded[:2], "big")
        return decoded[2: 2 + msg_len]

    def _enforce_pair_delta(self, dct_block: np.ndarray, a: Coord, b: Coord, bit: int) -> None:
        delta = self.config.delta
        diff = dct_block[a] - dct_block[b]
        target = delta if bit == 1 else -delta
        if bit == 1 and diff < target:
            adjust = (target - diff) / 2.0
            dct_block[a] += adjust
            dct_block[b] -= adjust
        elif bit == 0 and diff > target:
            adjust = (diff - target) / 2.0
            dct_block[a] -= adjust
            dct_block[b] += adjust

    def _load_image(self, path: pathlib.Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        return cv2.resize(img, (self.config.target_size, self.config.target_size))

    def _split_blocks(self, channel: np.ndarray) -> List[np.ndarray]:
        h, w = channel.shape
        bs = self.config.block_size
        blocks: List[np.ndarray] = []
        for y in range(0, h, bs):
            for x in range(0, w, bs):
                blocks.append(channel[y : y + bs, x : x + bs].copy())
        return blocks

    def _merge_blocks(self, blocks: Sequence[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        bs = self.config.block_size
        channel = np.zeros(shape, dtype=np.float32)
        idx = 0
        for y in range(0, h, bs):
            for x in range(0, w, bs):
                channel[y : y + bs, x : x + bs] = blocks[idx]
                idx += 1
        return channel

    def _capacity_bits(self, shape: Tuple[int, int]) -> int:
        h, w = shape
        blocks = (h // self.config.block_size) * (w // self.config.block_size)
        return blocks * len(self.coeff_pairs)

    def _build_payload(self, message: bytes) -> bytes:
        if len(message) > 0xFFFF:
            raise ValueError("Message too long (max 65535 bytes).")
        plaintext = len(message).to_bytes(2, "big") + message
        encoded = bytes(self.codec.encode(plaintext))
        header_plain = len(encoded).to_bytes(2, "big")
        header_encoded = bytes(self.codec.encode(header_plain))
        return header_encoded + encoded

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise DCT watermark embed/extract utility.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed = subparsers.add_parser("embed", help="Embed a secret message into an image.")
    embed.add_argument("--input", required=True, type=pathlib.Path, help="Path to source image.")
    embed.add_argument("--output", required=True, type=pathlib.Path, help="Path to write watermarked image.")
    embed.add_argument("--message", type=str, help="Secret message to embed.")
    embed.add_argument("--message-file", type=pathlib.Path, help="Path to file containing message bytes.")
    embed.add_argument("--pairs-per-block", type=int, default=14, help="Number of coefficient pairs per 8x8 block.")
    embed.add_argument("--ecc-symbols", type=int, default=96, help="Reed-Solomon parity symbols.")
    embed.add_argument("--delta", type=float, default=60.0, help="Minimum difference enforced between coefficient pairs.")
    embed.add_argument("--channel-index", type=int, default=1, choices=[0, 1, 2], help="BGR channel index to watermark (0=B,1=G,2=R).")
    embed.add_argument("--repeat-factor", type=int, default=11, help="Repeat each payload bit this many times (majority voting).")

    extract = subparsers.add_parser("extract", help="Extract the embedded secret message.")
    extract.add_argument("--input", required=True, type=pathlib.Path, help="Path to suspected watermarked image.")
    extract.add_argument("--output", type=pathlib.Path, help="Optional file to write recovered message bytes.")
    extract.add_argument("--pairs-per-block", type=int, default=14, help="Number of coefficient pairs (must match embed).")
    extract.add_argument("--ecc-symbols", type=int, default=96, help="Reed-Solomon parity symbols (must match embed).")
    extract.add_argument("--delta", type=float, default=60.0, help="Detection margin (must match embed).")
    extract.add_argument("--channel-index", type=int, default=1, choices=[0, 1, 2], help="Channel index used during embedding.")
    extract.add_argument("--repeat-factor", type=int, default=11, help="Repeat factor used during embedding.")

    return parser.parse_args()

def _load_message(args: argparse.Namespace) -> bytes:
    if args.message is not None:
        return args.message.encode("utf-8")
    if args.message_file is not None:
        return args.message_file.read_bytes()
    raise ValueError("Either --message or --message-file must be provided.")


def main() -> None:
    args = _parse_args()
    config = WatermarkConfig(
        pairs_per_block=args.pairs_per_block,
        ecc_symbols=args.ecc_symbols,
        delta=args.delta,
        channel_index=args.channel_index,
        repeat_factor=args.repeat_factor,
    )
    watermarker = DCTWatermarker(config)

    if args.command == "embed":
        message = _load_message(args)
        watermarker.embed(args.input, args.output, message)
    elif args.command == "extract":
        recovered = watermarker.extract(args.input)
        if args.output:
            args.output.write_bytes(recovered)
        else:
            print(recovered.decode("utf-8", errors="replace"))
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
