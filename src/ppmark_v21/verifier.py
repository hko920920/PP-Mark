from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .provider import ProviderRuntime
from .utils import load_image


@dataclass
class VerificationResult:
    is_valid: bool
    reason: str
    h_claimed: str
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
        claimed_hash = claimed.get("h_hex", "")
        claimed_payload = claimed.get("C_hex", "")
        claimed_sig = claimed.get("signature", "")
        proof_field = claimed.get("proof", "proof.bin")
        proof_path = Path(proof_field)
        if not proof_path.is_absolute():
            proof_path = (proof_dir / proof_path.name).resolve()
        public_inputs_path = proof_dir / "public_inputs.json"
        if not claimed_hash or not claimed_payload or not claimed_sig:
            return VerificationResult(
                is_valid=False,
                reason="missing-meta",
                h_claimed=claimed_hash,
                C_claimed=claimed_payload,
                C_extracted="",
                proof_ok=False,
                extractor_variant="",
            )
        if not proof_path.exists():
            return VerificationResult(
                is_valid=False,
                reason="missing-proof",
                h_claimed=claimed_hash,
                C_claimed=claimed_payload,
                C_extracted="",
                proof_ok=False,
                extractor_variant="",
            )
        proof_bytes = proof_path.read_bytes()
        if not self.runtime.signer.verify(claimed_hash, claimed_payload, proof_bytes, claimed_sig):
            return VerificationResult(
                is_valid=False,
                reason="signature-failed",
                h_claimed=claimed_hash,
                C_claimed=claimed_payload,
                C_extracted="",
                proof_ok=False,
                extractor_variant="",
            )
        if not public_inputs_path.exists():
            return VerificationResult(
                is_valid=False,
                reason="missing-public-inputs",
                h_claimed=claimed_hash,
                C_claimed=claimed_payload,
                C_extracted="",
                proof_ok=False,
                extractor_variant="",
            )
        public_inputs = self.runtime.halo2.read_public_inputs(public_inputs_path)
        anchor_pf = public_inputs.get("anchor_hex", "").lower().removeprefix("0x")
        poseidon_pf = public_inputs.get("poseidon_hex", "").lower().removeprefix("0x")
        if anchor_pf != claimed_hash.lower():
            return VerificationResult(
                is_valid=False,
                reason="anchor-mismatch",
                h_claimed=claimed_hash,
                C_claimed=claimed_payload,
                C_extracted="",
                proof_ok=False,
                extractor_variant="",
            )
        if poseidon_pf != claimed_payload.lower():
            return VerificationResult(
                is_valid=False,
                reason="payload-public-mismatch",
                h_claimed=claimed_hash,
                C_claimed=claimed_payload,
                C_extracted=poseidon_pf,
                proof_ok=False,
                extractor_variant="",
            )
        payload_bytes = bytes.fromhex(claimed_payload)
        recovered_hex = ""
        extractor_variant = ""
        meta_scheme = claimed.get("signature_scheme")
        if meta_scheme and meta_scheme.lower() != self.runtime.config.signature_scheme.lower():
            return VerificationResult(
                is_valid=False,
                reason="signature-scheme-mismatch",
                h_claimed=claimed_hash,
                C_claimed=claimed_payload,
                C_extracted="",
                proof_ok=False,
                extractor_variant="",
            )
        meta_pub = claimed.get("signature_pubkey_hex")
        runtime_pub = self.runtime.config.signing_pubkey_hex
        if runtime_pub and meta_pub and runtime_pub.lower() != meta_pub.lower():
            return VerificationResult(
                is_valid=False,
                reason="signature-pubkey-mismatch",
                h_claimed=claimed_hash,
                C_claimed=claimed_payload,
                C_extracted="",
                proof_ok=False,
                extractor_variant="",
            )
        # L1 attempt
        try:
            preprocessed = self.runtime.robin.prepare_for_extraction(image, payload_bytes)
        except Exception:
            preprocessed = image
        try:
            extraction = self.runtime.extractor.extract(preprocessed, payload_bytes=len(payload_bytes))
            recovered_bytes = self.runtime.extractor.bits_to_payload_bytes(extraction.bits)
            recovered_hex = recovered_bytes.hex()
            extractor_variant = extraction.variant
        except Exception:
            recovered_hex = ""
        # L2 fallback (pattern correlation) if L1 failed or mismatched.
        if recovered_hex != claimed_payload:
            tried_l2 = False
            try:
                bits = self.runtime.robin.extract_signal(image, payload_bits=len(payload_bytes) * 8)
                recovered_bytes = self.runtime.extractor.bits_to_payload_bytes(bits[: len(payload_bytes) * 8])
                recovered_hex = recovered_bytes.hex()
                extractor_variant = extractor_variant + "+l2corr" if extractor_variant else "l2corr"
                tried_l2 = True
            except Exception:
                pass
            # Frequency-domain fallback
            if recovered_hex != claimed_payload:
                try:
                    bits = self.runtime.robin.extract_signal_freq(image, payload_bits=len(payload_bytes) * 8)
                    recovered_bytes = self.runtime.extractor.bits_to_payload_bytes(bits[: len(payload_bytes) * 8])
                    recovered_hex = recovered_bytes.hex()
                    extractor_variant = extractor_variant + "+l2freq" if extractor_variant else "l2freq"
                    tried_l2 = True
                except Exception:
                    pass
        if recovered_hex != claimed_payload:
            return VerificationResult(
                is_valid=False,
                reason="payload-mismatch",
                h_claimed=claimed_hash,
                C_claimed=claimed_payload,
                C_extracted=recovered_hex,
                proof_ok=False,
                extractor_variant=extractor_variant,
            )
        proof_ok = self.runtime.halo2.verify(proof_dir)
        return VerificationResult(
            is_valid=proof_ok,
            reason="ok" if proof_ok else "proof-failed",
            h_claimed=claimed_hash,
            C_claimed=claimed_payload,
            C_extracted=recovered_hex,
            proof_ok=proof_ok,
            extractor_variant=extractor_variant,
        )
