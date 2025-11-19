#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import torch
    import lpips

    _LPIPS_AVAILABLE = True
except Exception:  # pragma: no cover
    _LPIPS_AVAILABLE = False


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_metrics(original: np.ndarray, compared: np.ndarray) -> Dict[str, float]:
    if original.shape != compared.shape:
        raise ValueError("Original and compared images must have the same shape.")
    metrics: Dict[str, float] = {}
    metrics["psnr"] = float(
        peak_signal_noise_ratio(original, compared, data_range=255)
    )
    metrics["ssim"] = float(
        structural_similarity(original, compared, multichannel=True, data_range=255)
    )
    if _LPIPS_AVAILABLE:
        loss_fn = lpips.LPIPS(net="vgg").eval()
        with torch.no_grad():
            tensor_a = (
                torch.from_numpy(original)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            tensor_b = (
                torch.from_numpy(compared)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            lp = loss_fn(tensor_a, tensor_b)
            metrics["lpips_vgg"] = float(lp.item())
    else:
        metrics["lpips_vgg"] = float("nan")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute image quality metrics (PSNR, SSIM, optional LPIPS) between base and watermarked images."
    )
    parser.add_argument("--original", type=Path, required=True, help="Path to the original/base image.")
    parser.add_argument("--compared", type=Path, required=True, help="Path to the watermarked/attacked image.")
    parser.add_argument(
        "--out", type=Path, default=None, help="Optional JSON path to write metric results."
    )
    args = parser.parse_args()

    original = load_image(args.original)
    compared = load_image(args.compared)
    metrics = compute_metrics(original, compared)

    if args.out:
        import json

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"[quality] Saved metrics to {args.out}")
    else:
        for key, value in metrics.items():
            print(f"{key}: {value}")
    if not _LPIPS_AVAILABLE:
        print("[quality] lpips package not installed; LPIPS metric reported as NaN.")


if __name__ == "__main__":
    main()
