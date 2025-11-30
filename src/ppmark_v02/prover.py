from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .config import ProviderConfig
from .crypto import poseidon_payload
from .hashing import compute_phash
from .payload import bytes_to_bits
from .provider import ProviderRuntime
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
        output_dir: Path,
        *,
        base_image: Path | None = None,
        noise_seed: int = 0,
    ) -> ProverOutputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        base_path = base_image or self.runtime.config.default_base_image
        if base_path is None:
            raise ValueError("No base image supplied and config.default_base_image missing.")
        base = load_image(base_path)
        phash = compute_phash(base)
        payload = poseidon_payload(self.runtime.secret, phash)
        patterned = self.runtime.pattern.apply(base, payload)
        latent = self.runtime.litevae.encode(patterned)
        payload_bits = bytes_to_bits(payload)
        watermarked = self.runtime.litevae.decode(latent, payload_bits)
        # Channel-2 sanity check
        extraction = self.runtime.extractor.extract(watermarked, payload_bytes=len(payload))
        extracted_payload = self.runtime.extractor.bits_to_payload_bytes(extraction.bits)
        proof_path, public_inputs_path = self.runtime.halo2.prove(
            secret=self.runtime.secret,
            anchor=phash,
            out_dir=output_dir,
        )
        image_path = output_dir / "watermarked.png"
        save_image(image_path, watermarked)
        metadata = self._build_metadata(
            prompt=prompt,
            noise_seed=noise_seed,
            phash=phash,
            payload=payload,
            extraction_variant=extraction.variant,
            extraction_matches=(extracted_payload == payload),
            proof_path=proof_path,
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
        prompt: str,
        noise_seed: int,
        phash: bytes,
        payload: bytes,
        extraction_variant: str,
        extraction_matches: bool,
        proof_path: Path,
    ) -> Dict[str, str]:
        return {
            "prompt": prompt,
            "noise_seed": str(noise_seed),
            "h_hex": phash.hex(),
            "C_hex": payload.hex(),
            "extractor_variant": extraction_variant,
            "channel2_status": "ok" if extraction_matches else "mismatch",
            "proof": str(proof_path),
        }
