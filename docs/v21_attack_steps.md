# PP-Mark v2.1 Attack Steps (ready-to-run)

## Scope / assumptions
- Minimal metadata: `h_hex`, `C_hex`, `proof`, plus `signature = Sign(sha256(h‖C‖proof))`.
- ZKP binds `h` as public input; Halo2 verify fails if meta is tampered or mismatched.
- L2 extractor is heuristic (correlation + freq fallback); regen/Zhao defense is partial.
- C2PA/chain anchoring optional (not baseline); treat as add-on only.

## CPU profile (already implemented)
Script: `scripts/run_attack_matrix_v21.py`
Targets: clean, jpeg_q60, gaussian_sigma3, center_crop_0.9, down_up_scale, hflip, vflip (+ optional rotate ±15° & elastic with --stress)
Run:
```bash
python scripts/run_attack_matrix_v21.py \
  --config artifacts/provider_setup_v21.json \
  --image results/v21_demo/watermarked.png \
  --metadata results/v21_demo/metadata.json \
  [--stress]
```
Output: `attack_matrix_v21.json` next to metadata.

## GPU/Heavy profiles (to implement/execute on server)
- **Zhao-type regeneration**: `scripts/run_regen_surrogate_v21.py` (CPU surrogate) added; `scripts/run_regen_attack_v21.py` wraps WatermarkAttacker+diffusers (GPU, external repo path required).
- **Rotation/Elastic**: stress profile partially added (--stress: rotate ±15°, elastic warp placeholder). Stronger versions/angles can be added.
- **Müller positioning/reprompt**: launcher `scripts/run_mueller_attack.py` (requires `external/semantic-forgery` checkout + GPU); adapt outputs to v2.1 when executing.

## What is ready vs. TODO
- Ready: CPU attack harness; regen wrapper script present (needs external WatermarkAttacker + GPU).
- TODO (server/GPU): hook regeneration outputs into eval pipeline, add rotation/elastic into attack_suite, and execute Müller positioning script. Execution should be done on GPU/Jupyter server.
