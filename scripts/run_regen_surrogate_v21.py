#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict

import cv2
import numpy as np

from ppmark_v21.config import ProviderConfig
from ppmark_v21.provider import ProviderRuntime
from ppmark_v21.utils import load_image

AttackFn = Callable[[np.ndarray], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Surrogate Zhao-style regeneration attacks (CPU-friendly) for v2.1.")
    parser.add_argument("--config", type=Path, required=True, help="Provider config JSON.")
    parser.add_argument("--image", type=Path, required=True, help="Watermarked image path.")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata JSON containing C_hex.")
    return parser.parse_args()


def regen_suite() -> Dict[str, AttackFn]:
    return {
        "blur_sigma5": lambda img: cv2.GaussianBlur(img, (7, 7), sigmaX=5.0),
        "jpeg_q40": lambda img: _jpeg(img, 40),
        "down_up_0.5": lambda img: _down_up(img, 0.5),
        "noise_std10": lambda img: _gaussian_noise(img, std=10.0),
        "combo_noise_blur": lambda img: _jpeg(_gaussian_noise(cv2.GaussianBlur(img, (5, 5), 3.0), 8.0), 50),
    }


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    _, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _down_up(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    nh = max(4, int(h * scale))
    nw = max(4, int(w * scale))
    down = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)


def _gaussian_noise(image: np.ndarray, std: float) -> np.ndarray:
    rng = np.random.default_rng(1337)
    noise = rng.normal(0, std, size=image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def main() -> None:
    args = parse_args()
    provider_cfg = ProviderConfig.load(args.config)
    runtime = ProviderRuntime(provider_cfg)
    with args.metadata.open("r", encoding="utf-8") as handle:
        claimed = json.load(handle)
    payload_bytes = bytes.fromhex(claimed["C_hex"])
    payload_len = len(payload_bytes)
    image = load_image(args.image)
    results = {}
    for name, fn in regen_suite().items():
        attacked = fn(image)
        variant = "n/a"
        try:
            report = runtime.extractor.extract(attacked, payload_bytes=payload_len)
            variant = report.variant
            recovered = runtime.extractor.bits_to_payload_bytes(report.bits)
            valid = recovered == payload_bytes
        except Exception as exc:  # pylint: disable=broad-except
            valid = False
            print(f"[regen] {name} failed: {exc}")
        results[name] = {"success": valid, "variant": variant}
        print(f"[regen] {name} success={valid}")
    out_path = args.metadata.parent / "regen_surrogate_v21.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[regen] Saved summary to {out_path}")


if __name__ == "__main__":
    main()
