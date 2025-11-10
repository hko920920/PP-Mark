#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_IMG="$ROOT_DIR/assets/processed/coco/coco_val_street_market.png"
WORKDIR="$ROOT_DIR/results"
MESSAGE=${1:-"PP-Mark Rocks"}

python3 "$ROOT_DIR/scripts/prepare_samples.py"
mkdir -p "$WORKDIR"
python3 "$ROOT_DIR/dct_watermark.py" embed --input "$ASSET_IMG" --output "$WORKDIR/watermarked.png" --message "$MESSAGE"
python3 "$ROOT_DIR/attack_matrix.py" --input "$ASSET_IMG" --workdir "$WORKDIR" --message "$MESSAGE"

echo "\nFinished local pipeline. Watermarked image and attack outputs are under $WORKDIR" 
