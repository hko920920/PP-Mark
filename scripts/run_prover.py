#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from ppmark_v02.config import ProviderConfig
from ppmark_v02.provider import ProviderRuntime
from ppmark_v02.prover import ProverService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PP-Mark v0.2 prover pipeline.")
    parser.add_argument("--config", type=Path, required=True, help="Provider config JSON.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the base generation.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store outputs.")
    parser.add_argument("--base-image", type=Path, help="Override default base image.")
    parser.add_argument("--noise-seed", type=int, default=0, help="Noise seed forwarded to the metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProviderConfig.load(args.config)
    runtime = ProviderRuntime(config)
    service = ProverService(runtime)
    outputs = service.generate(
        prompt=args.prompt,
        output_dir=args.output_dir,
        base_image=args.base_image,
        noise_seed=args.noise_seed,
    )
    print(f"Watermarked image: {outputs.image_path}")
    print(f"Metadata: {outputs.metadata_path}")
    print(f"Proof: {outputs.proof_path}")
    print(f"Public inputs: {outputs.public_inputs_path}")


if __name__ == "__main__":
    main()
