from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import cv2
import numpy as np

from .configs import DataConfig

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class ImageSample:
    image: np.ndarray
    path: Path
    index: int


class ImageStream:
    def __init__(self, paths: Sequence[Path], config: DataConfig):
        if not paths:
            raise ValueError("No training images found.")
        self.paths = list(paths)
        self.config = config
        self.batch_size = max(1, config.batch_size)
        self.num_samples = min(len(self.paths), config.max_samples or len(self.paths))
        self.rng = np.random.default_rng(config.seed)

    def iter_epoch(self) -> Iterator[List[ImageSample]]:
        indices = np.arange(len(self.paths))
        if self.config.shuffle:
            self.rng.shuffle(indices)
        usable = indices[: self.num_samples]
        for start in range(0, len(usable), self.batch_size):
            batch_idx = usable[start : start + self.batch_size]
            samples: List[ImageSample] = []
            for idx in batch_idx:
                image = _load_and_augment(self.paths[idx], self.rng, self.config)
                samples.append(ImageSample(image=image, path=self.paths[idx], index=idx))
            yield samples


def build_image_stream(config: DataConfig) -> ImageStream:
    files: List[Path] = []
    for root in config.resolved_roots():
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() in SUPPORTED_EXTS:
                files.append(path)
    files.sort()
    return ImageStream(files, config)


def _load_and_augment(path: Path, rng: np.random.Generator, config: DataConfig) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load {path}")
    aug = image.copy()
    aug = _random_crop(aug, rng, config.crop_ratio_range)
    if config.horizontal_flip and rng.random() < 0.5:
        aug = cv2.flip(aug, 1)
    aug = _jpeg_perturb(aug, rng, config.jpeg_quality_range)
    aug = _color_jitter(aug, rng, config.color_jitter)
    return aug


def _random_crop(image: np.ndarray, rng: np.random.Generator, ratio_range: tuple[float, float]) -> np.ndarray:
    lo, hi = ratio_range
    lo = max(0.5, min(lo, 1.0))
    hi = max(lo, min(hi, 1.0))
    ratio = rng.uniform(lo, hi)
    if ratio >= 0.999:
        return image
    h, w = image.shape[:2]
    nh = max(8, int(h * ratio))
    nw = max(8, int(w * ratio))
    sy = (h - nh) // 2
    sx = (w - nw) // 2
    cropped = image[sy : sy + nh, sx : sx + nw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def _jpeg_perturb(image: np.ndarray, rng: np.random.Generator, quality_range: tuple[int, int]) -> np.ndarray:
    ql, qh = quality_range
    ql = max(10, min(ql, 100))
    qh = max(ql, min(qh, 100))
    quality = int(rng.uniform(ql, qh))
    success, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return image
    decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if decoded is None:
        return image
    return decoded


def _color_jitter(image: np.ndarray, rng: np.random.Generator, jitter: float) -> np.ndarray:
    if jitter <= 0:
        return image
    noise = rng.normal(0.0, jitter, size=image.shape).astype(np.float32)
    adjusted = np.clip(image.astype(np.float32) / 255.0 + noise, 0.0, 1.0)
    return (adjusted * 255.0).astype(np.uint8)
