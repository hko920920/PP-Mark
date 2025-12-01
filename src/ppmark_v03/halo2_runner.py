"""Halo2 prover/verifier scaffolding."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .config import ZKConfig
from .halo2_interface import Halo2Package, PublicInputs, Witness
from .sampling import SampleObservation, SampleSet


@dataclass(slots=True)
class Halo2Paths:
    public: Path
    witness: Path
    proof: Path


def _serialize_public_inputs(public_inputs: PublicInputs, sample_count: int) -> Dict[str, str | int]:
    return {
        "prompt_hash": hex(public_inputs.prompt_hash),
        "binding": hex(public_inputs.binding),
        "sample_merkle_root": public_inputs.sample_merkle_root.hex(),
        "sample_count": sample_count,
    }


def _serialize_observation(obs: SampleObservation) -> Dict[str, float | int]:
    return {
        "index": obs.index,
        "uniform": obs.uniform,
        "z_expected": obs.z_expected,
        "z_observed": obs.z_observed,
    }


def _serialize_witness(witness: Witness) -> Dict[str, object]:
    return {
        "secret_key": witness.secret_key.hex(),
        "seed": witness.seed.hex(),
        "codeword": witness.codeword.hex(),
        "sample_observations": [_serialize_observation(obs) for obs in witness.sample_observations],
    }


def prepare_prover_package(package: Halo2Package, sample_set: SampleSet, output_dir: Path) -> Halo2Paths:
    output_dir.mkdir(parents=True, exist_ok=True)
    public_path = output_dir / "public_inputs.json"
    witness_path = output_dir / "witness.json"
    proof_path = package.proof_path or (output_dir / "proof.bin")

    public_payload = _serialize_public_inputs(package.public_inputs, len(sample_set.indices))
    with public_path.open("w", encoding="utf-8") as handle:
        json.dump(public_payload, handle, indent=2)

    witness_payload = _serialize_witness(package.witness)
    with witness_path.open("w", encoding="utf-8") as handle:
        json.dump(witness_payload, handle, indent=2)

    return Halo2Paths(public=public_path, witness=witness_path, proof=proof_path)


def _run_command(cmd: str, env: Dict[str, str]) -> None:
    args = shlex.split(cmd)
    subprocess.run(args, check=True, env=env)


def run_halo2_prover(paths: Halo2Paths, zk_config: ZKConfig) -> None:
    env = os.environ.copy()
    env.update(
        {
            "HALO2_PUBLIC": str(paths.public),
            "HALO2_WITNESS": str(paths.witness),
            "HALO2_PROOF": str(paths.proof),
        }
    )
    if zk_config.prover_cmd:
        _run_command(zk_config.prover_cmd, env)
    else:
        paths.proof.parent.mkdir(parents=True, exist_ok=True)
        with paths.proof.open("wb") as handle:
            handle.write(b"HALO2_STUB")
            handle.write(int(time.time()).to_bytes(8, "big", signed=False))


def verify_halo2_proof(paths: Halo2Paths, zk_config: ZKConfig) -> None:
    env = os.environ.copy()
    env.update(
        {
            "HALO2_PUBLIC": str(paths.public),
            "HALO2_PROOF": str(paths.proof),
        }
    )
    if zk_config.verifier_cmd:
        _run_command(zk_config.verifier_cmd, env)
    else:
        if not paths.proof.exists():
            raise FileNotFoundError(f"Proof file missing: {paths.proof}")
