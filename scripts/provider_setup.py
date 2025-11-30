#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from ppmark_v02.provider import create_provider_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PP-Mark v0.2 provider assets.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/provider_setup.json"),
        help="Path to the provider config JSON that will be created.",
    )
    parser.add_argument(
        "--halo2-dir",
        type=Path,
        default=Path("halo2_prover"),
        help="Path to the Halo2 prover workspace.",
    )
    parser.add_argument(
        "--base-image",
        type=Path,
        default=Path("assets/processed/coco/coco_val_street_market.png"),
        help="Default base image used in the demo pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = create_provider_config(
        output_path=args.out,
        halo2_dir=args.halo2_dir,
        base_image=args.base_image,
    )
    print(f"Wrote provider config to {artifacts.config_path}")
    print(f"Master secret hex: {artifacts.secret_hex}")


if __name__ == "__main__":
    main()
