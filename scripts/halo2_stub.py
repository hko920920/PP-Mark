#!/usr/bin/env python3
"""Placeholder Halo2 prover/verifier executable."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prove(public_path: Path, witness_path: Path, proof_path: Path) -> None:
    public = _load_json(public_path)
    witness = _load_json(witness_path)
    if len(witness.get("sample_observations", [])) != public.get("sample_count"):
        raise RuntimeError("Sample count mismatch between witness and public inputs")
    digest = hashlib.sha256()
    digest.update(json.dumps(public, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(witness, sort_keys=True).encode("utf-8"))
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    with proof_path.open("wb") as handle:
        handle.write(digest.digest())
    print(f"halo2_stub: wrote proof to {proof_path}")


def verify(public_path: Path, proof_path: Path) -> None:
    if not proof_path.exists():
        raise FileNotFoundError(f"Proof missing: {proof_path}")
    public = _load_json(public_path)
    digest = hashlib.sha256()
    digest.update(json.dumps(public, sort_keys=True).encode("utf-8"))
    proof = proof_path.read_bytes()
    if len(proof) != digest.digest_size:
        raise RuntimeError("Stub proof length mismatch")
    print("halo2_stub: verification passed (stub)")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"prove", "verify"}:
        raise SystemExit("Usage: halo2_stub.py [prove|verify]")
    public = Path(os.environ.get("HALO2_PUBLIC", ""))
    if not public:
        raise SystemExit("HALO2_PUBLIC env var is required")
    proof = Path(os.environ.get("HALO2_PROOF", ""))
    if sys.argv[1] == "prove":
        witness = Path(os.environ.get("HALO2_WITNESS", ""))
        if not witness:
            raise SystemExit("HALO2_WITNESS env var is required for proving")
        prove(public, witness, proof)
    else:
        verify(public, proof)


if __name__ == "__main__":
    main()
