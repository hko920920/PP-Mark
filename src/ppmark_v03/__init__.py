"""PP-Mark v0.3 core package."""

from .config import GlobalConfig, ImageConfig, ModelConfig, TableConfig, WatermarkConfig, ZKConfig
from .crypto import FIELD_MODULUS, bind_payload, hash_prompt_to_field, poseidon_hash_bytes, shuffle_seeded_indices
from .keys import KeyPair, generate_keypair, generate_secret_scalar, derive_public_point
from .payload import PayloadArtifacts, build_payload
from .rs import ReedSolomonCodec
from .sampling import SampleObservation, SampleSet, SampleTrace, deterministic_sample
from .tables import InverseCDFTable
from .embedding import EmbeddingResult, inject_watermark
from .cuda import DeviceConfig, get_watermark_kernel
from .halo2_interface import PublicInputs, Witness, Halo2Package, build_sample_commitment
from .halo2_runner import Halo2Paths, prepare_prover_package, run_halo2_prover, verify_halo2_proof

__all__ = [
    "GlobalConfig",
    "ImageConfig",
    "ModelConfig",
    "TableConfig",
    "WatermarkConfig",
    "ZKConfig",
    "FIELD_MODULUS",
    "bind_payload",
    "hash_prompt_to_field",
    "poseidon_hash_bytes",
    "shuffle_seeded_indices",
    "KeyPair",
    "generate_keypair",
    "generate_secret_scalar",
    "derive_public_point",
    "PayloadArtifacts",
    "build_payload",
    "ReedSolomonCodec",
    "SampleObservation",
    "SampleSet",
    "SampleTrace",
    "deterministic_sample",
    "InverseCDFTable",
    "EmbeddingResult",
    "inject_watermark",
    "DeviceConfig",
    "get_watermark_kernel",
    "PublicInputs",
    "Witness",
    "Halo2Package",
    "Halo2Paths",
    "build_sample_commitment",
    "prepare_prover_package",
    "run_halo2_prover",
    "verify_halo2_proof",
]
