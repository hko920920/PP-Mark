#!/usr/bin/env python3
"""Run diffusion-based regeneration attacks using external/WatermarkAttacker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
WMA_ROOT = ROOT / "external" / "WatermarkAttacker"
if str(WMA_ROOT) not in sys.path:
    sys.path.append(str(WMA_ROOT))

from regen_pipe import ReSDPipeline  # type: ignore  # pylint: disable=wrong-import-position
from wmattacker import DiffWMAttacker  # type: ignore  # pylint: disable=wrong-import-position


PRECISION_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusion regeneration attack (GPU only)")
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "watermarked.png",
                        help="Watermarked image to attack (auto-resized to 512x512).")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "regen_attack.png",
                        help="Where to store the regenerated output image.")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Hugging Face diffusion checkpoint ID.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device (cuda, cuda:1, etc.).")
    parser.add_argument("--precision", choices=PRECISION_MAP.keys(), default="fp16",
                        help="Torch dtype used for the pipeline.")
    parser.add_argument("--noise-step", type=int, default=60,
                        help="Diffusion noise step (higher => stronger attack).")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for regeneration (keep small unless VRAM abundant).")
    parser.add_argument("--prompt", type=str, default="",
                        help="Optional prompt to guide regeneration (empty string uses unconditional).")
    parser.add_argument("--enable-xformers", action="store_true",
                        help="Call pipe.enable_xformers_memory_efficient_attention().")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input image not found: {args.input}")
    if not WMA_ROOT.exists():
        raise SystemExit(f"WatermarkAttacker repo not found at {WMA_ROOT}")

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
    attacker.attack([str(args.input)], [str(args.output)])
    print(f"[pp_mark] Diffusion regeneration output saved to {args.output}")


if __name__ == "__main__":
    main()
