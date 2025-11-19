#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from ppmark_v21.config import ProviderConfig
from ppmark_v21.litevae import LiteVAE
from ppmark_v21.utils import load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search LiteVAE conditioning strengths.")
    parser.add_argument("--config", type=Path, required=True, help="Provider config JSON.")
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        required=True,
        help="Image root used for evaluation (can be passed multiple times).",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/litevae_strength.json"), help="JSON summary path.")
    parser.add_argument(
        "--strength",
        type=float,
        action="append",
        help="Candidate conditioning strength. Pass multiple times to override defaults.",
    )
    parser.add_argument("--max-samples", type=int, default=8, help="Maximum number of images evaluated.")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed for payload bits.")
    return parser.parse_args()


def collect_images(roots: List[Path]) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(p for p in root.rglob("*") if p.suffix.lower() in exts)
    files.sort()
    return files


def evaluate(cfg: ProviderConfig, images: List[Path], strengths: List[float], seed: int) -> dict:
    litevae = LiteVAE(cfg.litevae)
    rng = np.random.default_rng(seed)
    payload_bits = cfg.extractor.payload_bits
    scores: dict[float, float] = {}
    for strength in strengths:
        litevae.config.conditioning_strength = float(strength)
        metrics: List[float] = []
        for path in images:
            image = load_image(path)
            latent = litevae.encode(image)
            bits = rng.integers(0, 2, size=payload_bits, dtype=np.uint8)
            decoded = litevae.decode(latent, bits)
            diff = np.mean(np.abs(decoded.astype(np.float32) - image.astype(np.float32)))
            metrics.append(float(diff))
        scores[strength] = float(np.mean(metrics))
    return scores


def main() -> None:
    args = parse_args()
    provider_cfg = ProviderConfig.load(args.config)
    images = collect_images([root.resolve() for root in args.data_root])[: args.max_samples]
    if not images:
        raise RuntimeError("No evaluation images found.")
    strengths = args.strength or [0.05, 0.08, 0.1, 0.12, 0.15]
    scores = evaluate(provider_cfg, images, strengths, args.seed)
    best_strength = min(scores, key=scores.get)
    payload = {
        "conditioning_strengths": {str(k): v for k, v in scores.items()},
        "best": {
            "conditioning_strength": best_strength,
            "score": scores[best_strength],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[litevae] Saved sweep summary to {args.output}")


if __name__ == "__main__":
    main()
