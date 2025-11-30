#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from ppmark_v02.config import PatternConfig, ProviderConfig
from ppmark_v02.patterns import SemanticPatternInjector
from ppmark_v02.utils import load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select PatternGen amplitudes/frequencies.")
    parser.add_argument("--config", type=Path, required=True, help="Provider config JSON.")
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        required=True,
        help="Image root used for evaluation (can be provided multiple times).",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/pattern_selection.json"), help="JSON summary path.")
    parser.add_argument("--amplitude", type=float, action="append", help="Candidate amplitude.")
    parser.add_argument("--frequency", type=int, action="append", help="Candidate radial frequency.")
    parser.add_argument("--max-samples", type=int, default=6, help="Maximum number of images evaluated.")
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


def evaluate(cfg: ProviderConfig, images: List[Path], amplitudes: List[float], freqs: List[int], seed: int) -> List[dict]:
    rng = np.random.default_rng(seed)
    payload_bytes = rng.bytes(32)
    results: List[dict] = []
    for amp in amplitudes:
        for freq in freqs:
            local_cfg = PatternConfig(**cfg.pattern.__dict__)
            local_cfg.amplitude = float(amp)
            local_cfg.radial_frequency = int(freq)
            injector = SemanticPatternInjector(local_cfg)
            diffs: List[float] = []
            for path in images:
                image = load_image(path)
                marked = injector.apply(image, payload_bytes)
                diff = float(np.mean(np.abs(marked.astype(np.float32) - image.astype(np.float32))))
                diffs.append(diff)
            avg_diff = float(np.mean(diffs))
            score = float(amp) / (1.0 + avg_diff)
            results.append({
                "amplitude": float(amp),
                "radial_frequency": int(freq),
                "avg_difference": avg_diff,
                "score": score,
            })
    return results


def main() -> None:
    args = parse_args()
    provider_cfg = ProviderConfig.load(args.config)
    images = collect_images([root.resolve() for root in args.data_root])[: args.max_samples]
    if not images:
        raise RuntimeError("No evaluation images found.")
    amplitudes = args.amplitude or [0.05, 0.07, 0.09]
    frequencies = args.frequency or [10, 12, 16]
    grid = evaluate(provider_cfg, images, amplitudes, frequencies, args.seed)
    best = max(grid, key=lambda item: item["score"])
    payload = {"best_params": {"amplitude": best["amplitude"], "radial_frequency": best["radial_frequency"]}, "grid": grid}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[pattern] Saved selection summary to {args.output}")


if __name__ == "__main__":
    main()
