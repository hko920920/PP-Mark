from __future__ import annotations

import numpy as np


def bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.ndim != 1:
        raise ValueError("bits must be a flat array.")
    pad = (-len(bits)) % 8
    if pad:
        bits = np.pad(bits, (0, pad))
    grouped = bits.reshape(-1, 8)
    packed = np.packbits(grouped, axis=1)
    return bytes(packed.flatten())


def repeat_bits(bits: np.ndarray, repeats: int) -> np.ndarray:
    if repeats <= 1:
        return bits.copy()
    return np.repeat(bits, repeats)


def majority_vote(bits: np.ndarray, repeats: int) -> np.ndarray:
    if repeats <= 1:
        return bits.copy()
    usable = (len(bits) // repeats) * repeats
    bits = bits[:usable]
    grouped = bits.reshape(-1, repeats)
    threshold = (repeats // 2) + 1
    return (grouped.sum(axis=1) >= threshold).astype(np.uint8)
