#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_FORGERY = ROOT / "external" / "semantic-forgery"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Müller et al. Imprint-Forgery attack")
    parser.add_argument("--cover-image", type=Path, default=ROOT / "results" / "watermarked.png")
    parser.add_argument("--cover-mask", type=Path, default=None)
    parser.add_argument("--target-prompt", type=str, default="cyberpunk skyline at blue hour, ultra detailed")
    parser.add_argument("--wm-type", choices=["GS", "TR"], default="GS")
    parser.add_argument("--target-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--attacker-model", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--steps", type=int, default=151)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "mueller_imprint")
    parser.add_argument("--extra", nargs=argparse.REMAINDER)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        "python",
        "run_imprint_forgery.py",
        "--wm_type",
        args.wm_type,
        "--cover_image_path",
        str(args.cover_image),
        "--target_prompt",
        args.target_prompt,
        "--modelid_target",
        args.target_model,
        "--modelid_attacker",
        args.attacker_model,
        "--steps",
        str(args.steps),
        "--out_dir",
        str(args.out_dir),
    ]
    if args.cover_mask:
        cmd += ["--cover_mask_path", str(args.cover_mask)]
    if args.extra:
        cmd += args.extra
    return cmd

def main() -> None:
    args = parse_args()
    if not SEMANTIC_FORGERY.exists():
        raise SystemExit(f"semantic-forgery repo not found under {SEMANTIC_FORGERY}")
    if not args.cover_image.exists():
        raise SystemExit(f"cover image not found: {args.cover_image}")
    cmd = build_command(args)
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[pp_mark] Executing in {SEMANTIC_FORGERY}:\n  {printable}")
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=SEMANTIC_FORGERY, check=True)

if __name__ == "__main__":
    main()
