"""Poseidon helpers and deterministic samplers."""

from __future__ import annotations

import hashlib
import random
from typing import List, Sequence

FIELD_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617


class PoseidonUnavailableError(RuntimeError):
    """Raised when poseidon bindings are not installed."""


def hash_prompt_to_field(prompt: str) -> int:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % FIELD_MODULUS


def poseidon_hash_elements(elements: Sequence[int]) -> int:
    try:
        from poseidon_py.poseidon_hash import poseidon_hash  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise PoseidonUnavailableError(
            "poseidon_py is required for Poseidon hashing. Install poseidon-py."
        ) from exc

    iterator = iter(elements)
    try:
        acc = next(iterator) % FIELD_MODULUS
    except StopIteration as exc:  # pragma: no cover
        raise ValueError("poseidon_hash_elements requires a non-empty sequence") from exc
    for elem in iterator:
        acc = poseidon_hash(acc, elem % FIELD_MODULUS) % FIELD_MODULUS
    return acc % FIELD_MODULUS


def poseidon_hash_bytes(payload: bytes) -> bytes:
    ints = [int.from_bytes(payload[i : i + 31], "big") % FIELD_MODULUS for i in range(0, len(payload), 31)]
    digest = poseidon_hash_elements(ints)
    return digest.to_bytes(32, "big")


def bind_payload(prompt: str, seed: bytes, secret: bytes) -> int:
    if not seed or not secret:
        raise ValueError("Seed and secret must be non-empty")
    h_p = hash_prompt_to_field(prompt)
    seed_field = int.from_bytes(seed, "big") % FIELD_MODULUS
    secret_field = int.from_bytes(secret, "big") % FIELD_MODULUS
    return poseidon_hash_elements([h_p, seed_field, secret_field])


def shuffle_seeded_indices(length: int, key_material: bytes, sample_count: int) -> List[int]:
    if length <= 0:
        raise ValueError("length must be positive")
    if sample_count <= 0 or sample_count > length:
        raise ValueError("sample_count must be in [1, length]")
    digest = hashlib.blake2s(key_material).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    indices = list(range(length))
    rng.shuffle(indices)
    return indices[:sample_count]
