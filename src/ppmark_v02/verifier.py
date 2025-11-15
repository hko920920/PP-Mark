from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .hashing import compute_phash
from .provider import ProviderRuntime
from .utils import load_image


@dataclass
class VerificationResult:
    is_valid: bool
    reason: str
    h_claimed: str
    h_extracted: str
    C_claimed: str
    C_extracted: str
    proof_ok: bool
    extractor_variant: str


class VerifierService:
    def __init__(self, runtime: ProviderRuntime):
        self.runtime = runtime

    def verify(
        self,
        image_path: Path,
        proof_dir: Path,
        claimed: Dict[str, str],
    ) -> VerificationResult:
        image = load_image(image_path)
        extracted_hash = compute_phash(image).hex()
        claimed_hash = claimed.get("h_hex", "")
        claimed_payload = claimed.get("C_hex", "")
        payload_bytes = bytes.fromhex(claimed_payload) if claimed_payload else b""
        proof_ok = False
        if not payload_bytes:
            return VerificationResult(
                is_valid=False,
                reason="missing-payload",
                h_claimed=claimed_hash,
                h_extracted=extracted_hash,
                C_claimed=claimed_payload,
                C_extracted="",
                proof_ok=False,
                extractor_variant="",
            )
        extraction = self.runtime.extractor.extract(image, payload_bytes=len(payload_bytes))
        recovered_bytes = self.runtime.extractor.bits_to_payload_bytes(extraction.bits)
        recovered_hex = recovered_bytes.hex()
        if claimed_hash != extracted_hash:
            return VerificationResult(
                is_valid=False,
                reason="phash-mismatch",
                h_claimed=claimed_hash,
                h_extracted=extracted_hash,
                C_claimed=claimed_payload,
                C_extracted=recovered_hex,
                proof_ok=False,
                extractor_variant=extraction.variant,
            )
        if recovered_hex != claimed_payload:
            return VerificationResult(
                is_valid=False,
                reason="payload-mismatch",
                h_claimed=claimed_hash,
                h_extracted=extracted_hash,
                C_claimed=claimed_payload,
                C_extracted=recovered_hex,
                proof_ok=False,
                extractor_variant=extraction.variant,
            )
        proof_ok = self.runtime.halo2.verify(proof_dir)
        return VerificationResult(
            is_valid=proof_ok,
            reason="ok" if proof_ok else "proof-failed",
            h_claimed=claimed_hash,
            h_extracted=extracted_hash,
            C_claimed=claimed_payload,
            C_extracted=recovered_hex,
            proof_ok=proof_ok,
            extractor_variant=extraction.variant,
        )
