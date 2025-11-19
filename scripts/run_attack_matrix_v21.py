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
    parser = argparse.ArgumentParser(description="Evaluate v2.1 extractor robustness against basic attacks (optional stress profile).")
    parser.add_argument("--config", type=Path, required=True, help="Provider config JSON.")
    parser.add_argument("--image", type=Path, required=True, help="Clean/watermarked image path.")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata JSON containing C_hex.")
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Include stress attacks (rotate ±15, elastic warp placeholder).",
    )
    return parser.parse_args()


def attack_suite(include_stress: bool = False) -> Dict[str, AttackFn]:
    attacks: Dict[str, AttackFn] = {
        "clean": lambda img: img,
        "jpeg_q60": lambda img: _jpeg_attack(img, 60),
        "gaussian_sigma3": lambda img: cv2.GaussianBlur(img, (5, 5), sigmaX=3.0),
        "center_crop_0.9": lambda img: _center_crop(img, 0.9),
        "down_up_scale": lambda img: _down_up(img, 0.7),
        "hflip": lambda img: cv2.flip(img, 1),
        "vflip": lambda img: cv2.flip(img, 0),
    }
    if include_stress:
        attacks.update(
            {
                "rotate_p15": lambda img: _rotate(img, 15),
                "rotate_n15": lambda img: _rotate(img, -15),
                "elastic": lambda img: _elastic_warp(img, alpha=8, sigma=4),
            }
        )
    return attacks


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


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated


def _down_up(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    nh = max(4, int(h * scale))
    nw = max(4, int(w * scale))
    down = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)


def _elastic_warp(image: np.ndarray, alpha: float = 5, sigma: float = 3) -> np.ndarray:
    # Lightweight elastic warp adapted for CPU (placeholder; not as strong as torch-based versions).
    rng = np.random.default_rng(1337)
    shape = image.shape[:2]
    dx = cv2.GaussianBlur((rng.random(shape) * 2 - 1).astype(np.float32), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((rng.random(shape) * 2 - 1).astype(np.float32), (17, 17), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


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
    for name, fn in attack_suite(args.stress).items():
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
    out_path = args.metadata.parent / "attack_matrix_v21.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[attack] Saved summary to {out_path}")


if __name__ == "__main__":
    main()
