from __future__ import annotations

import secrets
from typing import Iterable


def generate_master_secret(num_bytes: int = 32) -> bytes:
    if num_bytes <= 0:
        raise ValueError("Master secret length must be positive.")
    return secrets.token_bytes(num_bytes)


def _poseidon_hash(data: Iterable[int]) -> bytes:
    try:
        from poseidon_py.poseidon_hash import poseidon_hash_many  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("poseidon-py is required for Poseidon hashing.") from exc

    field_elements = list(data)
    digest = poseidon_hash_many(field_elements)
    return int(digest).to_bytes(32, "big")


def poseidon_payload(secret: bytes, anchor_hash: bytes) -> bytes:
    if not secret or not anchor_hash:
        raise ValueError("Secret and anchor hash must be non-empty.")
    secret_int = int.from_bytes(secret, "big")
    anchor_int = int.from_bytes(anchor_hash, "big")
    return _poseidon_hash([secret_int, anchor_int])
