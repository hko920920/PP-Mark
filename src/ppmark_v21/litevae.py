from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pywt

from .config import LiteVAEConfig
from .payload import repeat_bits


@dataclass
class WaveletLatent:
    coeffs: List[List[np.ndarray]]


class LiteVAE:
    def __init__(self, config: LiteVAEConfig):
        self.config = config
        self.wavelet = pywt.Wavelet(config.wavelet)

    def encode(self, image: np.ndarray) -> WaveletLatent:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("LiteVAE expects HxWx3 images.")
        coeffs: List[List[np.ndarray]] = []
        for c in range(3):
            channel = image[:, :, c].astype(np.float32) / 255.0
            coeff = pywt.wavedec2(channel, self.wavelet, level=self.config.levels)
            coeffs.append(coeff)
        return WaveletLatent(coeffs=coeffs)

    def decode(
        self,
        latent: WaveletLatent,
        payload_bits: np.ndarray,
        *,
        repeats: int | None = None,
        conditioning_strength: float | None = None,
    ) -> np.ndarray:
        conditioned_bits = repeat_bits(payload_bits, repeats or self.config.payload_repeats)
        rng = np.random.default_rng(self.config.carrier_seed)
        restored_channels: List[np.ndarray] = []
        for coeff in latent.coeffs:
            conditioned = self._inject_payload(coeff, conditioned_bits, rng, conditioning_strength)
            restored = pywt.waverec2(conditioned, self.wavelet)
            restored = np.clip(restored, 0.0, 1.0)
            restored_channels.append(np.rint(restored * 255.0).astype(np.uint8))
        stacked = np.stack(restored_channels, axis=2)
        return stacked

    def _inject_payload(
        self,
        coeffs: List[np.ndarray],
        bits: np.ndarray,
        rng: np.random.Generator,
        conditioning_strength: float | None = None,
    ) -> List[np.ndarray]:
        conditioned = [coeff.copy() if isinstance(coeff, np.ndarray) else list(coeff) for coeff in coeffs]
        if len(conditioned) < 2:
            return conditioned
        detail = conditioned[-1]  # type: ignore[index]
        if not isinstance(detail, (list, tuple)):
            return conditioned
        diag_band = detail[2]
        flat = diag_band.flatten()
        if len(flat) < len(bits):
            raise ValueError("Payload larger than LiteVAE diagonal capacity.")
        idx = rng.choice(len(flat), size=len(bits), replace=False)
        idx.sort()
        strength = conditioning_strength if conditioning_strength is not None else self.config.conditioning_strength
        signed = (bits * 2 - 1).astype(np.float32)
        flat[idx] = (1.0 - strength) * flat[idx] + strength * signed
        detail = (detail[0], detail[1], flat.reshape(diag_band.shape))
        conditioned[-1] = detail  # type: ignore[index]
        return conditioned
