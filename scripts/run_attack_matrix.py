#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict

import cv2
import numpy as np

from ppmark_v02.config import ProviderConfig
from ppmark_v02.provider import ProviderRuntime
from ppmark_v02.utils import load_image

AttackFn = Callable[[np.ndarray], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate extractor robustness against basic attacks.")
    parser.add_argument("--config", type=Path, required=True, help="Provider config JSON.")
    parser.add_argument("--image", type=Path, required=True, help="Clean/watermarked image path.")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata JSON containing C_hex.")
    return parser.parse_args()


def attack_suite() -> Dict[str, AttackFn]:
    return {
        "clean": lambda img: img,
        "jpeg_q60": lambda img: _jpeg_attack(img, 60),
        "gaussian_sigma3": lambda img: cv2.GaussianBlur(img, (5, 5), sigmaX=3.0),
        "center_crop_0.9": lambda img: _center_crop(img, 0.9),
    }


def _jpeg_attack(image: np.ndarray, quality: int) -> np.ndarray:
    _, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _center_crop(image: np.ndarray, ratio: float) -> np.ndarray:
    h, w = image.shape[:2]
    nh = int(h * ratio)
    nw = int(w * ratio)
    sy = (h - nh) // 2
    sx = (w - nw) // 2
    cropped = image[sy : sy + nh, sx : sx + nw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


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
    for name, fn in attack_suite().items():
        attacked = fn(image)
        variant = "n/a"
        try:
            report = runtime.extractor.extract(attacked, payload_bytes=payload_len)
            variant = report.variant
            recovered = runtime.extractor.bits_to_payload_bytes(report.bits)
            valid = recovered == payload_bytes
        except Exception as exc:  # pylint: disable=broad-except
            valid = False
            print(f"[attack] {name} failed: {exc}")
        results[name] = {"success": valid, "variant": variant}
        print(f"[attack] {name} success={valid}")
    out_path = args.metadata.parent / "attack_matrix.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[attack] Saved summary to {out_path}")


if __name__ == "__main__":
    main()
