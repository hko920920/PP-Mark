from __future__ import annotations

import cv2
import numpy as np

from .config import PatternConfig
from .payload import bytes_to_bits


class SemanticPatternInjector:
    def __init__(self, config: PatternConfig):
        self.config = config

    def apply(self, image: np.ndarray, payload: bytes) -> np.ndarray:
        if not self.config.enabled:
            return image
        bits = bytes_to_bits(payload)
        pattern = self._generate_pattern(bits, image.shape[:2])
        pattern01 = 0.5 + 0.5 * pattern
        adjusted = image.astype(np.float32) / 255.0
        blended = (1.0 - self.config.amplitude) * adjusted + self.config.amplitude * pattern01
        blended = np.clip(blended, 0.0, 1.0)
        return np.rint(blended * 255.0).astype(np.uint8)

    def remove(self, image: np.ndarray, payload: bytes) -> np.ndarray:
        """패턴을 선형적으로 제거하여 L1 추출 안정화."""
        if not self.config.enabled:
            return image
        bits = bytes_to_bits(payload)
        pattern = self._generate_pattern(bits, image.shape[:2])
        pattern01 = 0.5 + 0.5 * pattern
        adjusted = image.astype(np.float32) / 255.0
        denom = max(1e-4, 1.0 - self.config.amplitude)
        restored = (adjusted - self.config.amplitude * pattern01) / denom
        restored = np.clip(restored, 0.0, 1.0)
        return np.rint(restored * 255.0).astype(np.uint8)

    def extract_signal(self, image: np.ndarray, payload_bits: int) -> np.ndarray:
        """
        L2 fallback: multi-scale, multi-angle correlation with reference pattern to recover bits.
        Still heuristic, but more robust than a single-scale projection.
        """
        if not self.config.enabled:
            raise ValueError("Pattern injector disabled; no L2 signal available.")
        h, w = image.shape[:2]
        ref_bits = np.zeros(payload_bits, dtype=np.uint8)

        def correlate(img_arr: np.ndarray, pattern: np.ndarray) -> np.ndarray:
            pattern = pattern - np.mean(pattern)
            img_norm = img_arr - np.mean(img_arr)
            corr_map = np.mean(img_norm * pattern, axis=2)
            flat = corr_map.flatten()
            thr = np.median(flat)
            bits_local = (flat > thr).astype(np.uint8)
            if len(bits_local) < payload_bits:
                bits_local = np.pad(bits_local, (0, payload_bits - len(bits_local)))
            return bits_local[:payload_bits]

        img_f = image.astype(np.float32) / 255.0
        patterns = []
        # base scale
        patterns.append(self._generate_pattern(ref_bits, (h, w)))
        # downscale/blur for robustness
        half = cv2.resize(img_f, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        patterns.append(self._generate_pattern(ref_bits, (h // 2, w // 2)))
        # slight rotations
        center = (w // 2, h // 2)
        for angle in (-10, 0, 10):
            mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_f, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            patterns.append((rotated, self._generate_pattern(ref_bits, (h, w))))

        votes = []
        # First pattern at original scale
        votes.append(correlate(img_f, patterns[0]))
        # Downscale pattern correlates against resized image
        votes.append(correlate(half, patterns[1]))
        # Rotated variants
        for rotated, pat in patterns[2:]:
            votes.append(correlate(rotated, pat))

        stacked = np.stack(votes, axis=0)
        majority = (stacked.sum(axis=0) >= (stacked.shape[0] // 2 + 1)).astype(np.uint8)
        return majority

    def extract_signal_freq(self, image: np.ndarray, payload_bits: int) -> np.ndarray:
        """
        Alternative L2 fallback: simple frequency-domain magnitude thresholding.
        Treats watermark as low/mid-frequency bias. Heuristic; no guarantees.
        """
        import cv2  # local import to avoid hard dependency at import time

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        fft = np.fft.fft2(gray)
        mag = np.abs(fft)
        # Focus on low/mid frequencies
        h, w = mag.shape
        lh, lw = h // 2, w // 2
        window = mag[:lh, :lw]
        flat = window.flatten()
        # Normalize
        flat = flat - np.mean(flat)
        thr = np.median(flat)
        bits = (flat > thr).astype(np.uint8)
        if len(bits) < payload_bits:
            bits = np.pad(bits, (0, payload_bits - len(bits)))
        return bits[:payload_bits]

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
