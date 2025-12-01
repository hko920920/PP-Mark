"""Reference watermark embedding pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import GlobalConfig
from .noise import NoiseArtifacts, generate_noise_artifacts
from .sampling import SampleSet, SampleTrace
from .tables import InverseCDFTable


@dataclass(slots=True)
class EmbeddingResult:
    latent_noise: np.ndarray
    noise_artifacts: NoiseArtifacts
    sample_trace: SampleTrace


def inject_watermark(
    binding: int,
    bit_sequence: Sequence[int],
    config: GlobalConfig,
    inverse_cdf: InverseCDFTable,
    sample_set: SampleSet,
) -> EmbeddingResult:
    total_pixels = config.image.total_pixels
    artifacts = generate_noise_artifacts(
        binding=binding,
        bit_sequence=bit_sequence,
        total_pixels=total_pixels,
        inverse_cdf=inverse_cdf,
        alpha=config.image.alpha,
    )
    sample_trace = SampleTrace()
    for idx in sample_set.indices:
        sample_trace.record(
            idx,
            float(artifacts.uniforms[idx]),
            float(artifacts.gaussian[idx]),
            float(artifacts.combined[idx]),
        )
    latent_noise = artifacts.combined.reshape(config.image.height, config.image.width)
    return EmbeddingResult(latent_noise=latent_noise, noise_artifacts=artifacts, sample_trace=sample_trace)
