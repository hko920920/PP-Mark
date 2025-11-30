#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppmark_v02.config import ProviderConfig
from ppmark_v02.provider import ProviderRuntime
from ppmark_v02.verifier import VerifierService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PP-Mark v0.2 verifier pipeline.")
    parser.add_argument("--config", type=Path, required=True, help="Provider config JSON.")
    parser.add_argument("--image", type=Path, required=True, help="Image to verify.")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata JSON containing h/C.")
    parser.add_argument(
        "--proof-dir",
        type=Path,
        help="Directory containing proof.bin + public_inputs.json. Defaults to metadata directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proof_dir = args.proof_dir or args.metadata.parent
    with args.metadata.open("r", encoding="utf-8") as handle:
        claimed = json.load(handle)
    config = ProviderConfig.load(args.config)
    runtime = ProviderRuntime(config)
    service = VerifierService(runtime)
    result = service.verify(args.image, proof_dir, claimed)
    print(f"Verification status: {result.reason}")
    print(f"h_claimed={result.h_claimed}")
    print(f"h_extracted={result.h_extracted}")
    print(f"C_claimed={result.C_claimed}")
    print(f"C_extracted={result.C_extracted}")
    print(f"Extractor variant={result.extractor_variant}")
    print(f"Proof OK? {result.proof_ok}")
    if not result.is_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
