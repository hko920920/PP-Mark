"""CUDA integration scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .config import GlobalConfig
from .embedding import EmbeddingResult, inject_watermark
from .noise import generate_noise_artifacts
from .sampling import SampleSet, SampleTrace
from .tables import InverseCDFTable

try:  # pragma: no cover - optional dependency
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None


@dataclass(slots=True)
class DeviceConfig:
    backend: str = "cpu"
    device_id: int = 0
    stream: int | None = None
    log_buffer: str | None = None


class WatermarkKernel(Protocol):
    def embed(
        self,
        binding: int,
        bit_sequence: Sequence[int],
        sample_set: SampleSet,
    ) -> EmbeddingResult: ...


class CPUWatermarkKernel:
    def __init__(self, config: GlobalConfig, inverse_cdf: InverseCDFTable) -> None:
        self.config = config
        self.inverse_cdf = inverse_cdf

    def embed(
        self,
        binding: int,
        bit_sequence: Sequence[int],
        sample_set: SampleSet,
    ) -> EmbeddingResult:
        return inject_watermark(
            binding=binding,
            bit_sequence=bit_sequence,
            config=self.config,
            inverse_cdf=self.inverse_cdf,
            sample_set=sample_set,
        )


class CUDAWatermarkKernel:
    def __init__(self, config: GlobalConfig, inverse_cdf: InverseCDFTable) -> None:
        if cp is None:
            raise RuntimeError("CuPy is required for CUDA backend")
        self.config = config
        self.inverse_cdf = inverse_cdf
        self.inverse_table_gpu = cp.asarray(inverse_cdf.values)

    def embed(
        self,
        binding: int,
        bit_sequence: Sequence[int],
        sample_set: SampleSet,
    ) -> EmbeddingResult:
        if cp is None:
            raise RuntimeError("CUDA backend unavailable")
        total_pixels = self.config.image.total_pixels
        bits = np.array(bit_sequence, dtype=np.int8)
        gpu_bits = cp.asarray(bits)
        repeats = int(np.ceil(total_pixels / len(bits)))
        tiled = cp.tile(gpu_bits, repeats)[:total_pixels]
        spread = (tiled.astype(cp.float32) * 2.0 - 1.0).astype(cp.float32)
        uniforms = self._poseidon_uniforms(binding, total_pixels)
        gaussian = self._lookup_inverse_cdf(uniforms)
        combined = gaussian + self.config.image.alpha * spread
        cpu_combined = cp.asnumpy(combined)
        sample_trace = SampleTrace()
        for idx in sample_set.indices:
            sample_trace.record(
                idx,
                float(cp.asnumpy(uniforms[idx])),
                float(cp.asnumpy(gaussian[idx])),
                float(cpu_combined[idx]),
            )
        latent_noise = cpu_combined.reshape(self.config.image.height, self.config.image.width)
        artifacts = generate_noise_artifacts(
            binding=binding,
            bit_sequence=bit_sequence,
            total_pixels=total_pixels,
            inverse_cdf=self.inverse_cdf,
            alpha=self.config.image.alpha,
        )
        artifacts.combined[:] = cpu_combined.reshape(-1)  # align CPU artifacts with GPU output
        return EmbeddingResult(latent_noise=latent_noise, noise_artifacts=artifacts, sample_trace=sample_trace)

    def _poseidon_uniforms(self, binding: int, count: int) -> "cp.ndarray":
        from .crypto import poseidon_hash_elements

        uniforms = cp.empty(count, dtype=cp.float32)
        seed = binding
        for i in range(count):
            digest = poseidon_hash_elements([seed, i + 1])
            uniforms[i] = (digest % (1 << 32)) / float(1 << 32)
        return uniforms

    def _lookup_inverse_cdf(self, uniforms: "cp.ndarray") -> "cp.ndarray":
        idx = cp.clip((uniforms * (self.inverse_table_gpu.size - 1)).astype(cp.int64), 0, self.inverse_table_gpu.size - 1)
        return self.inverse_table_gpu[idx]


def get_watermark_kernel(
    device_cfg: DeviceConfig,
    config: GlobalConfig,
    inverse_cdf: InverseCDFTable,
) -> WatermarkKernel:
    if device_cfg.backend.lower() == "cuda":
        return CUDAWatermarkKernel(config, inverse_cdf)
    return CPUWatermarkKernel(config, inverse_cdf)
