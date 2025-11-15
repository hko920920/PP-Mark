from __future__ import annotations

import numpy as np

from .config import PatternConfig
from .payload import bytes_to_bits


class SemanticPatternInjector:
    def __init__(self, config: PatternConfig):
        self.config = config

    def apply(self, image: np.ndarray, payload: bytes) -> np.ndarray:
        bits = bytes_to_bits(payload)
        pattern = self._generate_pattern(bits, image.shape[:2])
        adjusted = image.astype(np.float32) / 255.0
        adjusted += self.config.amplitude * pattern
        adjusted = np.clip(adjusted, 0.0, 1.0)
        return (adjusted * 255.0).astype(np.uint8)

    def _generate_pattern(self, bits: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        y, x = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]
        radius = np.sqrt(x**2 + y**2)
        freq = self.config.radial_frequency
        tiled_bits = np.tile(bits, int(np.ceil(h * w / len(bits))))[: h * w]
        tiled_bits = tiled_bits.reshape(h, w)
        phase = np.pi * (2 * tiled_bits - 1)
        wave = np.sin(freq * radius + phase)
        pattern = np.repeat(wave[:, :, None], 3, axis=2)
        attenuation = 1.0 - np.clip(radius, 0.0, 1.0)
        pattern *= attenuation[:, :, None]
        return pattern.astype(np.float32)
