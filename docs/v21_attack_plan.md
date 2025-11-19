# PP-Mark v2.1 Attack Matrix Plan

## Assumptions / metadata
- Minimal meta: `h_hex`, `C_hex`, `proof`, `signature = Sign(sha256(h‖C‖proof))` (no prompt/seed/model_id/timestamp stored). Signature required; chain anchor (hash-only) optional.
- Seed/nonce: prefer 256-bit random hex; short seeds only for demos.

## Scripts
- `scripts/run_attack_matrix_v21.py`: basic JPEG/blur/center-crop harness (CPU).
- TODO GPU: Zhao-style regeneration, rotation/elastic, higher-stress profiles.

## How to run (CPU-friendly)
```bash
python scripts/run_attack_matrix_v21.py \
  --config artifacts/provider_setup_v21.json \
  --image results/v21_t2i/watermarked.png \
  --metadata results/v21_t2i/metadata.json
```
Outputs `attack_matrix_v21.json` under the metadata directory.

## GPU/Heavy profiles (to run on Jupyter/GPU server)
- **Regeneration (Zhao)**: `scripts/run_regen_attack_v21.py` wraps WatermarkAttacker (external repo) + diffusers; needs GPU. After attack, run verifier to log C extraction success.
- **Rotation/elastic**: extend attack_suite with rotations (±15°) and elastic warp; note expected failures and log separately (“stress” profile).
- **Müller imprint/reprompt**: requires proxy model + inversion; future work once GPU available.

## Status
- CPU attack harness ready (basic transforms).
- Regeneration harness present (needs WatermarkAttacker + GPU); Müller still pending; rotation/elastic stress to be added.
- Seed/nonce: prefer 256-bit random hex for anchors in production; short seeds are acceptable for demos but weak against brute force.
