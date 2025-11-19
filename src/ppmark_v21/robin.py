from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .config import PatternConfig
from .patterns import SemanticPatternInjector


@dataclass
class RobinTrace:
    t_injection: int
    guidance_scale: float
    hiding_prompt: str
    payload_checksum: str

    def to_dict(self) -> dict:
        return asdict(self)


class MidTrajectoryInjector:
    """Simulates ROBIN mid-trajectory injection on CPU-friendly latents."""

    def __init__(self, config: PatternConfig):
        self.config = config
        self.injector = SemanticPatternInjector(config)

    def inject(
        self,
        image: np.ndarray,
        payload: bytes,
        *,
        prompt: str,
        seed: str,
    ) -> tuple[np.ndarray, RobinTrace]:
        rng = np.random.default_rng(self._seed_from_inputs(seed, prompt))
        forward = self._forward_diffuse(image, rng)
        injected = self.injector.apply(forward, payload)
        guided = self._guided_reverse(image, injected)
        trace = RobinTrace(
            t_injection=self.config.t_injection,
            guidance_scale=self.config.guidance_scale,
            hiding_prompt=self.config.hiding_prompt,
            payload_checksum=hashlib.sha256(payload).hexdigest()[:32],
        )
        return guided, trace

    def prepare_for_extraction(self, image: np.ndarray, payload: bytes) -> np.ndarray:
        return self.injector.remove(image, payload)

    def extract_signal(self, image: np.ndarray, payload_bits: int) -> np.ndarray:
        return self.injector.extract_signal(image, payload_bits)

    def extract_signal_freq(self, image: np.ndarray, payload_bits: int) -> np.ndarray:
        return self.injector.extract_signal_freq(image, payload_bits)

    def _forward_diffuse(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        sigma = max(0.15, self.config.t_injection / 100.0)
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
        noise = rng.normal(0.0, sigma * 0.1, size=img.shape).astype(np.float32)
        degraded = np.clip(blurred + noise, 0.0, 1.0)
        return (degraded * 255.0).astype(np.uint8)

    def _guided_reverse(self, base: np.ndarray, injected: np.ndarray) -> np.ndarray:
        guidance = float(np.clip(self.config.guidance_scale, 0.05, 0.95))
        base_f = base.astype(np.float32)
        injected_f = injected.astype(np.float32)
        guided = (1.0 - guidance) * base_f + guidance * injected_f
        guided = np.clip(guided, 0.0, 255.0)
        return guided.astype(np.uint8)

    @staticmethod
    def _seed_from_inputs(seed: str, prompt: str) -> int:
        data = (seed + "|" + prompt).encode("utf-8")
        digest = hashlib.sha256(data).digest()
        return int.from_bytes(digest[:8], "big")
