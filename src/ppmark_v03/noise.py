"""Spread spectrum noise generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .tables import InverseCDFTable


@dataclass(slots=True)
class NoiseArtifacts:
    uniforms: np.ndarray
    gaussian: np.ndarray
    spread: np.ndarray
    combined: np.ndarray


def poseidon_uniforms(binding: int, count: int) -> np.ndarray:
    from .crypto import poseidon_hash_elements

    uniforms = np.empty(count, dtype=np.float32)
    seed = binding
    for i in range(count):
        digest = poseidon_hash_elements([seed, i + 1])
        uniforms[i] = (digest % (1 << 32)) / float(1 << 32)
    return uniforms


def spread_bits(bits: Sequence[int], total_pixels: int) -> np.ndarray:
    if len(bits) == 0:
        raise ValueError("Bit sequence must be non-empty")
    repeats = int(np.ceil(total_pixels / len(bits)))
    tiled = np.tile(bits, repeats)[:total_pixels]
    return (np.array(tiled, dtype=np.float32) * 2.0 - 1.0).astype(np.float32)


def generate_noise_artifacts(
    binding: int,
    bit_sequence: Sequence[int],
    total_pixels: int,
    inverse_cdf: InverseCDFTable,
    alpha: float,
) -> NoiseArtifacts:
    uniforms = poseidon_uniforms(binding, total_pixels)
    gaussian = inverse_cdf.batch_lookup(uniforms)
    spread = spread_bits(bit_sequence, total_pixels)
    combined = gaussian + alpha * spread
    return NoiseArtifacts(uniforms=uniforms, gaussian=gaussian, spread=spread, combined=combined)
