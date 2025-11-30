from __future__ import annotations

import hashlib
import secrets
from typing import Iterable


def generate_master_secret(num_bytes: int = 32) -> bytes:
    if num_bytes <= 0:
        raise ValueError("Master secret length must be positive.")
    return secrets.token_bytes(num_bytes)


def _poseidon_hash(data: Iterable[int]) -> bytes:
    try:
        from poseidon_py.poseidon_hash import poseidon_hash  # type: ignore

        field_elements = list(data)
        digest = poseidon_hash(field_elements)
        return int(digest).to_bytes(32, "big")
    except Exception:  # pragma: no cover - falls back for CPU-only envs
        m = hashlib.sha3_256()
        m.update(bytes(data))
        return m.digest()


def poseidon_payload(secret: bytes, anchor_hash: bytes) -> bytes:
    if not secret or not anchor_hash:
        raise ValueError("Secret and anchor hash must be non-empty.")
    merged = secret + anchor_hash
    int_stream = merged
    return _poseidon_hash(int_stream)
