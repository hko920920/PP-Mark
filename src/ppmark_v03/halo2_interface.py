"""Halo2 circuit interface definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .sampling import SampleObservation, SampleSet


@dataclass(slots=True)
class PublicInputs:
    prompt_hash: int
    binding: int
    sample_merkle_root: bytes


@dataclass(slots=True)
class Witness:
    secret_key: bytes
    seed: bytes
    codeword: bytes
    sample_observations: List[SampleObservation]


@dataclass(slots=True)
class Halo2Package:
    public_inputs: PublicInputs
    witness: Witness
    proof_path: Path | None = None


def build_sample_commitment(sample_set: SampleSet) -> Tuple[bytes, List[bytes]]:
    import hashlib

    data = b"".join(idx.to_bytes(4, "little") for idx in sample_set.indices)
    digest = hashlib.sha256(data).digest()
    return digest, []
