#!/usr/bin/env python3
"""
Zhao-style diffusion regeneration attack (GPU) for PP-Mark v2.1.

This is a thin wrapper around the WatermarkAttacker implementation from pp_mark/external/WatermarkAttacker.
It regenerates a watermarked image and saves the attacked output; verification/evaluation is done separately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusion regeneration attack (GPU, WatermarkAttacker).")
    parser.add_argument("--input", type=Path, required=True, help="Watermarked image to attack (auto-resized to 512x512).")
    parser.add_argument("--output", type=Path, required=True, help="Where to store the regenerated output image.")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face diffusion checkpoint ID.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (e.g., cuda, cuda:1).")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16", help="Torch dtype.")
    parser.add_argument("--noise-step", type=int, default=60, help="Diffusion noise step (higher => stronger attack).")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for regeneration.")
    parser.add_argument("--prompt", type=str, default="", help="Optional prompt for guided regeneration (empty = unconditional).")
    parser.add_argument(
        "--attacker-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "pp_mark" / "external" / "WatermarkAttacker",
        help="Path to WatermarkAttacker repo (with regen_pipe.py and wmattacker.py).",
    )
    parser.add_argument("--enable-xformers", action="store_true", help="Enable xformers memory efficient attention if available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input image not found: {args.input}")
    if not args.attacker_root.exists():
        raise SystemExit(f"WatermarkAttacker repo not found at {args.attacker_root} (set --attacker-root).")

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"torch not available: {exc}")

    # Wire up attacker deps
    if str(args.attacker_root) not in sys.path:
        sys.path.append(str(args.attacker_root))
    try:
        from regen_pipe import ReSDPipeline  # type: ignore  # pylint: disable=wrong-import-position
        from wmattacker import DiffWMAttacker  # type: ignore  # pylint: disable=wrong-import-position
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Failed to import WatermarkAttacker modules: {exc}")

    PRECISION_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = PRECISION_MAP[args.precision]

    pipe = ReSDPipeline.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        safety_checker=None,
        feature_extractor=None,
    )
    pipe = pipe.to(args.device)
    if args.enable_xformers and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        pipe.enable_xformers_memory_efficient_attention()

    captions = {}
    if args.prompt:
        captions[args.input.stem] = args.prompt

    attacker = DiffWMAttacker(pipe, batch_size=args.batch_size, noise_step=args.noise_step, captions=captions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    attacker.attack([str(args.input)], [str(args.output)])
    print(f"[regen] Diffusion regeneration output saved to {args.output}")


if __name__ == "__main__":
    main()
