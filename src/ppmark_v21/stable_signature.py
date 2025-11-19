from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .litevae import LiteVAE
from .payload import bytes_to_bits


class ExtractorProtocol(Protocol):
    def extract(self, image: np.ndarray, payload_bytes: int):
        ...

    def bits_to_payload_bytes(self, bits: np.ndarray) -> bytes:
        ...


@dataclass
class SignatureReport:
    variant: str
    iterations: int
    strength: float


class StableSignatureEncoder:
    """LiteVAE 기반 Stable Signature 삽입 루프.

    LiteVAE 디코더를 반복 호출하며 추출기가 항상 동일한 payload를 검출하도록 보정한다.
    """

    def __init__(
        self,
        litevae: LiteVAE,
        extractor: ExtractorProtocol,
        *,
        max_iters: int = 6,
        strength_step: float = 1.5,
        max_strength: float = 1.0,
    ):
        self.litevae = litevae
        self.extractor = extractor
        self.max_iters = max_iters
        self.strength_step = strength_step
        self.max_strength = max_strength

    def embed(
        self,
        image: np.ndarray,
        payload: bytes,
        *,
        postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
        pre_extract: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> tuple[np.ndarray, SignatureReport]:
        payload_bits = bytes_to_bits(payload)
        latent = self.litevae.encode(image)
        target_bytes = payload
        strength = self.litevae.config.conditioning_strength

        for iteration in range(1, self.max_iters + 1):
            candidate = self.litevae.decode(latent, payload_bits, conditioning_strength=strength)
            final_image = postprocess(candidate) if postprocess else candidate
            extract_image = pre_extract(final_image) if pre_extract else final_image
            extraction = self.extractor.extract(extract_image, payload_bytes=len(payload))
            recovered = self.extractor.bits_to_payload_bytes(extraction.bits)
            if recovered == target_bytes:
                return final_image, SignatureReport(
                    variant=extraction.variant,
                    iterations=iteration,
                    strength=strength,
                )
            strength = min(self.max_strength, strength * self.strength_step)

        # Final forced pass with maximum strength
        candidate = self.litevae.decode(latent, payload_bits, conditioning_strength=self.max_strength)
        final_image = postprocess(candidate) if postprocess else candidate
        extract_image = pre_extract(final_image) if pre_extract else final_image
        extraction = self.extractor.extract(extract_image, payload_bytes=len(payload))
        recovered = self.extractor.bits_to_payload_bytes(extraction.bits)
        if recovered == target_bytes:
            return final_image, SignatureReport(
                variant=extraction.variant,
                iterations=self.max_iters + 1,
                strength=self.max_strength,
            )

        raise RuntimeError("StableSignatureEncoder failed to converge; extractor mismatch.")
