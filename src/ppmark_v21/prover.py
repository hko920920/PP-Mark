from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .crypto import poseidon_payload
from .provider import ProviderRuntime
from .robin import RobinTrace
from .semantic_anchor import compute_semantic_anchor
from .utils import dump_json, load_image, save_image


@dataclass
class ProverOutputs:
    image_path: Path
    proof_path: Path
    public_inputs_path: Path
    metadata_path: Path


class ProverService:
    def __init__(self, runtime: ProviderRuntime):
        self.runtime = runtime

    def generate(
        self,
        prompt: str,
        seed: str,
        model_id: str,
        timestamp: str,
        output_dir: Path,
        *,
        parent_manifest_hash: Optional[str] = None,
        base_image: Path | None = None,
    ) -> ProverOutputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        base_path = base_image or self.runtime.config.default_base_image
        if base_path is None:
            raise ValueError("No base image supplied and config.default_base_image missing.")
        base = load_image(base_path)
        anchor = compute_semantic_anchor(prompt, seed, model_id, timestamp, parent_manifest_hash)
        payload = poseidon_payload(self.runtime.secret, anchor)
        robin_image, robin_trace = self.runtime.robin.inject(
            base,
            payload,
            prompt=prompt,
            seed=seed,
        )
        watermarked, signature_report = self.runtime.signature_encoder.embed(
            robin_image,
            payload,
            postprocess=lambda x: x,
            pre_extract=lambda img: self.runtime.robin.prepare_for_extraction(img, payload),
        )
        proof_path, public_inputs_path = self.runtime.halo2.prove(
            secret=self.runtime.secret,
            anchor=anchor,
            payload_hex=payload.hex(),
            out_dir=output_dir,
        )
        image_path = output_dir / "watermarked.png"
        save_image(image_path, watermarked)
        proof_bytes = proof_path.read_bytes()
        metadata = self._build_metadata(
            anchor=anchor,
            payload=payload,
            proof_path=proof_path,
            signature_report=signature_report,
            robin_trace=robin_trace,
            proof_bytes=proof_bytes,
        )
        metadata_path = output_dir / "metadata.json"
        dump_json(metadata_path, metadata)
        return ProverOutputs(
            image_path=image_path,
            proof_path=proof_path,
            public_inputs_path=public_inputs_path,
            metadata_path=metadata_path,
        )

    def _build_metadata(
        self,
        *,
        anchor: bytes,
        payload: bytes,
        proof_path: Path,
        signature_report,
        robin_trace: RobinTrace,
        proof_bytes: bytes,
    ) -> Dict[str, str]:
        signature = self.runtime.signer.sign(anchor, payload, proof_bytes)
        data = {
            "h_hex": anchor.hex(),
            "C_hex": payload.hex(),
            "proof": str(proof_path),
            "signature": signature,
            "signature_scheme": self.runtime.config.signature_scheme,
        }
        if signature_report:
            data["l1_variant"] = signature_report.variant
            data["l1_iterations"] = signature_report.iterations
            data["l1_strength"] = signature_report.strength
        if robin_trace:
            data["l2_trace"] = robin_trace.to_dict()
        pub_hex = self.runtime.signer.public_key_hex
        if pub_hex:
            data["signature_pubkey_hex"] = pub_hex
        return data
