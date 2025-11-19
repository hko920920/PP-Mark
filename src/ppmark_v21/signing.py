from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Optional, Protocol

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


class Signer(Protocol):
    scheme: str

    def sign(self, anchor: bytes, payload: bytes, proof_bytes: bytes) -> str: ...

    def verify(self, anchor_hex: str, payload_hex: str, proof_bytes: bytes, signature_hex: str) -> bool: ...

    @property
    def public_key_hex(self) -> str | None: ...


@dataclass
class HMACSigner:
    key: bytes
    hash_name: str = "sha256"
    scheme: str = "hmac-sha256"

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Signing key must be non-empty for HMAC signer.")

    @property
    def public_key_hex(self) -> str | None:
        return None

    def sign(self, anchor: bytes, payload: bytes, proof_bytes: bytes) -> str:
        digest = hashlib.sha256(anchor + payload + proof_bytes).digest()
        return hmac.new(self.key, digest, getattr(hashlib, self.hash_name)).hexdigest()

    def verify(self, anchor_hex: str, payload_hex: str, proof_bytes: bytes, signature_hex: str) -> bool:
        if not signature_hex:
            return False
        try:
            anchor = bytes.fromhex(anchor_hex)
            payload = bytes.fromhex(payload_hex)
        except ValueError:
            return False
        digest = hashlib.sha256(anchor + payload + proof_bytes).digest()
        expected = hmac.new(self.key, digest, getattr(hashlib, self.hash_name)).hexdigest()
        return hmac.compare_digest(expected, signature_hex.lower())


@dataclass
class ECDSASigner:
    private_key_der: Optional[bytes]
    public_key_der: bytes | None
    scheme: str = "ecdsa-p256"

    def __post_init__(self) -> None:
        if self.public_key_der is None and self.private_key_der is None:
            raise ValueError("ECDSA signer requires at least a public key.")
        self._priv = None
        self._pub = None

    def _load_private(self):
        if self._priv is None:
            if self.private_key_der is None:
                raise ValueError("ECDSA signer has no private key for signing.")
            self._priv = serialization.load_der_private_key(self.private_key_der, password=None)
        return self._priv

    def _load_public(self):
        if self._pub is None:
            if self.public_key_der is not None:
                self._pub = serialization.load_der_public_key(self.public_key_der)
            elif self._priv is not None:
                self._pub = self._priv.public_key()
            else:
                raise ValueError("ECDSA signer missing public key.")
        return self._pub

    @property
    def public_key_hex(self) -> str | None:
        if self.public_key_der is None and self._priv is not None:
            self.public_key_der = self._priv.public_key().public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        return self.public_key_der.hex() if self.public_key_der else None

    def sign(self, anchor: bytes, payload: bytes, proof_bytes: bytes) -> str:
        private_key = self._load_private()
        data = anchor + payload + proof_bytes
        signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
        return signature.hex()

    def verify(self, anchor_hex: str, payload_hex: str, proof_bytes: bytes, signature_hex: str) -> bool:
        if not signature_hex:
            return False
        try:
            anchor = bytes.fromhex(anchor_hex)
            payload = bytes.fromhex(payload_hex)
            signature = bytes.fromhex(signature_hex)
        except ValueError:
            return False
        public_key = self._load_public()
        data = anchor + payload + proof_bytes
        try:
            public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
            return True
        except Exception:
            return False


def create_signer(scheme: str, private_key: bytes | None, public_key: bytes | None) -> Signer:
    normalized = scheme.lower()
    if normalized in {"hmac", "hmac-sha256"}:
        if private_key is None:
            raise ValueError("HMAC signer requires a secret key.")
        return HMACSigner(key=private_key, hash_name="sha256", scheme="hmac-sha256")
    if normalized in {"ecdsa", "ecdsa-p256"}:
        if public_key is None and private_key is None:
            raise ValueError("ECDSA signer requires at least a public key.")
        return ECDSASigner(private_key_der=private_key, public_key_der=public_key, scheme="ecdsa-p256")
    raise ValueError(f"Unsupported signature scheme: {scheme}")
