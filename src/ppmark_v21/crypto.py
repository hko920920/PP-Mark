from __future__ import annotations

import hashlib
import secrets
from typing import Iterable, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def generate_master_secret(num_bytes: int = 32) -> bytes:
    if num_bytes <= 0:
        raise ValueError("Master secret length must be positive.")
    return secrets.token_bytes(num_bytes)


def generate_signing_key(num_bytes: int = 32) -> bytes:
    if num_bytes <= 0:
        raise ValueError("Signing key length must be positive.")
    return secrets.token_bytes(num_bytes)


def generate_ecdsa_keypair() -> Tuple[bytes, bytes]:
    private_key = ec.generate_private_key(ec.SECP256R1())
    priv_der = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_der, pub_der


def _poseidon_hash(data: Iterable[int]) -> bytes:
    try:
        from poseidon_py.poseidon_hash import poseidon_hash_many  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("poseidon-py is required for Poseidon hashing. Install extras or check virtualenv.") from exc

    field_elements = list(data)
    digest = poseidon_hash_many(field_elements)
    return int(digest).to_bytes(32, "big")


def poseidon_payload(secret: bytes, anchor: bytes) -> bytes:
    if not secret or not anchor:
        raise ValueError("Secret and anchor must be non-empty.")
    secret_int = int.from_bytes(secret, "big")
    anchor_int = int.from_bytes(anchor, "big")
    return _poseidon_hash([secret_int, anchor_int])
