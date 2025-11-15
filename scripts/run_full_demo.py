#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppmark_v02.config import ProviderConfig
from ppmark_v02.provider import ProviderRuntime, create_provider_config
from ppmark_v02.prover import ProverService
from ppmark_v02.verifier import VerifierService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run provider setup, prover, and verifier in one shot.")
    parser.add_argument("--config", type=Path, default=Path("artifacts/provider_setup.json"), help="Provider config JSON path.")
    parser.add_argument("--halo2-dir", type=Path, default=Path("halo2_prover"), help="Halo2 prover workspace.")
    parser.add_argument(
        "--base-image",
        type=Path,
        default=Path("assets/processed/coco/coco_val_street_market.png"),
        help="Default base image used for the demo run.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt recorded in metadata.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for prover outputs.")
    parser.add_argument("--noise-seed", type=int, default=0, help="Noise seed stored in metadata.")
    parser.add_argument("--skip-verify", action="store_true", help="Stop after the prover stage.")
    return parser.parse_args()


def ensure_config(path: Path, halo2_dir: Path, base_image: Path) -> ProviderConfig:
    if not path.exists():
        print(f"[demo] {path} missing — running Stage 1 setup.")
        create_provider_config(output_path=path, halo2_dir=halo2_dir, base_image=base_image)
    return ProviderConfig.load(path)


def main() -> None:
    args = parse_args()
    provider_cfg = ensure_config(args.config, args.halo2_dir, args.base_image)
    runtime = ProviderRuntime(provider_cfg)
    prover = ProverService(runtime)
    outputs = prover.generate(
        prompt=args.prompt,
        output_dir=args.output_dir,
        base_image=args.base_image,
        noise_seed=args.noise_seed,
    )
    print(f"[demo] Watermarked image: {outputs.image_path}")
    print(f"[demo] Metadata: {outputs.metadata_path}")
    print(f"[demo] Proof: {outputs.proof_path}")
    if args.skip_verify:
        return
    verifier = VerifierService(runtime)
    with outputs.metadata_path.open("r", encoding="utf-8") as handle:
        claimed = json.load(handle)
    result = verifier.verify(outputs.image_path, outputs.metadata_path.parent, claimed)
    print(f"[demo] Verification status: {result.reason}")
    if not result.is_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
