"""Payload construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .crypto import bind_payload, hash_prompt_to_field
from .rs import ReedSolomonCodec


@dataclass(slots=True)
class PayloadArtifacts:
    prompt_hash: int
    binding: int
    message_bytes: bytes
    codeword: bytes
    bits: List[int]


def int_to_fixed_bytes(value: int, length: int) -> bytes:
    data = value.to_bytes(length, "big")
    if len(data) > length:
        raise ValueError("Value exceeds fixed length")
    return data[-length:]


def bytes_to_bits(payload: bytes) -> List[int]:
    bits: List[int] = []
    for byte in payload:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def build_payload(prompt: str, seed: bytes, secret: bytes, codec: ReedSolomonCodec) -> PayloadArtifacts:
    if not seed or not secret:
        raise ValueError("Seed and secret must be non-empty")
    prompt_hash = hash_prompt_to_field(prompt)
    binding = bind_payload(prompt, seed, secret)
    message_bytes = int_to_fixed_bytes(binding, codec.k)
    codeword = codec.encode(message_bytes)
    bits = bytes_to_bits(codeword)
    return PayloadArtifacts(
        prompt_hash=prompt_hash,
        binding=binding,
        message_bytes=message_bytes,
        codeword=codeword,
        bits=bits,
    )
