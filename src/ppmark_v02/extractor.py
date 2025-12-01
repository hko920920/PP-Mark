from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import cv2
import numpy as np

from .config import ExtractorConfig
from .litevae import LiteVAE
from .models.extractor_net import NeuralExtractor
from .payload import bits_to_bytes as pack_bits, majority_vote


@dataclass
class ExtractionReport:
    bits: np.ndarray
    variant: str


class RobustExtractor:
    def __init__(self, litevae: LiteVAE, config: ExtractorConfig):
        self.litevae = litevae
        self.config = config

    def extract(self, image: np.ndarray, payload_bytes: int) -> ExtractionReport:
        bits_needed = payload_bytes * 8
        last_error: Exception | None = None
        for variant_name, variant in self._iter_variants(image):
            try:
                bits = self._extract_variant(variant, bits_needed)
                return ExtractionReport(bits=bits, variant=variant_name)
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
        raise ValueError("Extractor failed on all variants") from last_error

    def bits_to_payload_bytes(self, bits: np.ndarray) -> bytes:
        return pack_bits(bits)

    def _extract_variant(self, image: np.ndarray, bits_needed: int) -> np.ndarray:
        latent = self.litevae.encode(image)
        repeats = self.litevae.config.payload_repeats
        sample_len = bits_needed * repeats
        rng = np.random.default_rng(self.litevae.config.carrier_seed)
        votes: List[np.ndarray] = []
        for coeff in latent.coeffs:
            detail = coeff[-1]
            diag = detail[2].flatten()
            if len(diag) < sample_len:
                raise ValueError("LiteVAE detail band smaller than payload requirements.")
            idx = rng.choice(len(diag), size=sample_len, replace=False)
            picked = diag[idx]
            votes.append((picked >= 0).astype(np.uint8))
        stacked = np.stack(votes, axis=0)
        combined = (stacked.sum(axis=0) >= (stacked.shape[0] // 2 + 1)).astype(np.uint8)
        collapsed = majority_vote(combined, repeats)
        if len(collapsed) < bits_needed:
            raise ValueError("Collapsed bits shorter than requested payload.")
        return collapsed[:bits_needed]

    def _iter_variants(self, image: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
        yield "full", image
        if self.config.crop_ratio < 1.0:
            cropped = self._center_crop(image, self.config.crop_ratio)
            resized = cv2.resize(cropped, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            yield "center_crop", resized
        resized = cv2.resize(image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        yield "resize", resized

    @staticmethod
    def _center_crop(image: np.ndarray, ratio: float) -> np.ndarray:
        h, w = image.shape[:2]
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        return image[start_y:start_y + new_h, start_x:start_x + new_w]


class NeuralExtractorWrapper:
    """Adapter that exposes the same API as RobustExtractor but runs a neural extractor."""

    def __init__(self, neural: NeuralExtractor, payload_bits: int):
        self.neural = neural
        self.payload_bits = payload_bits

    def extract(self, image: np.ndarray, payload_bytes: int) -> ExtractionReport:
        bits = self.neural.predict_bits(image)
        required = payload_bytes * 8
        if len(bits) < required:
            raise ValueError("Neural extractor produced insufficient bits.")
        return ExtractionReport(bits=bits[:required], variant="neural")

    def bits_to_payload_bytes(self, bits: np.ndarray) -> bytes:
        return pack_bits(bits)
