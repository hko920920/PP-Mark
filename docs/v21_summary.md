# PP-Mark v2.1 Summary

## Core structure
- Semantic anchor `h = H(prompt, seed, model_id, timestamp, parent_hash)` (no pHash). Payload `C = Poseidon(k‖h)`.
- L1: LiteVAE + tiling extractor (geom/quality robustness). Learnable weights can be attached.
- L2: Mid-trajectory pattern injection (ROBIN style), heuristic fallback (multi-scale/rotation).
- L3: Halo2 circuit binds anchor as public input (Poseidon output + anchor rows). `public_inputs.json` annotated with `anchor_hex` when available.
- CLI: `ppmark-v21` (`setup/prover/verify`), one-shot demo `scripts/run_full_demo_v21.py`.
- Attack harness: `scripts/run_attack_matrix_v21.py` (clean/JPEG/blur/crop + optional stress: rotate/elastic).
- C2PA helper: `scripts/embed_c2pa_v21.py` (needs c2patool; unsigned by default, supports signing if cert/key provided).

## Docs / samples
- `docs/sample_t2i_metadata_v21.json`, `docs/sample_i2i_metadata_v21.json`: minimal samples with h_hex, C_hex, proof + signature (sig over sha256(h‖C‖proof), 256-bit nonces used).
- `docs/v21_attack_plan.md`, `docs/v21_attack_steps.md`, `docs/seed_guidance_v21.md` (256-bit nonce recommended).

## Tests
- Added: `tests/test_semantic_anchor.py`, `tests/test_prover_verifier_stub.py` (not run locally; use venv + `PYTHONPATH=src pytest -q`).

## TODO / caveats
- Metadata integrity: signature over (h‖C‖proof) is now required; verification flow should check signature first. Blockchain/anchor (hash-only) is optional/deferred for cost/ops reasons; treat it as an add-on, not required baseline.
- Attack benchmarks (regen/Zhao, rotation/elastic, Müller positioning) not implemented/run; GPU + custom scripts required.
- C2PA full JSON-LD/signing/verify chain not applied (helper only).
- L2 extractor remains heuristic; no learned L2 or diffusers-based regen attack implemented.
- Seed validation: 256-bit nonce recommended; no strict enforcement in code.
- Halo2 circuit updated but cargo prove/verify not executed in this environment (stub if cargo missing).
- Proof binding: public_inputs.json stores Poseidon(secret, anchor) and anchor as public inputs. At verify time, meta h_hex must match the anchor in public_inputs.json; otherwise verification fails.

## Quick run (local CPU)
```bash
# 1) setup
ppmark-v21 setup --out artifacts/provider_setup_v21.json \
  --halo2-dir halo2_prover \
  --base-image assets/processed/coco/coco_val_street_market.png

# 2) watermark
ppmark-v21 prover \
  --config artifacts/provider_setup_v21.json \
  --prompt "demo" \
  --seed 9f3b6c2f4e7d1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6 \
  --model-id "demo-model" \
  --timestamp "2025-11-18T12:10:30Z" \
  --output-dir results/v21_demo

# 3) verify
ppmark-v21 verify \
  --config artifacts/provider_setup_v21.json \
  --image results/v21_demo/watermarked.png \
  --metadata results/v21_demo/metadata.json

# 4) attacks (CPU)
python scripts/run_attack_matrix_v21.py \
  --config artifacts/provider_setup_v21.json \
  --image results/v21_demo/watermarked.png \
  --metadata results/v21_demo/metadata.json \
  --stress

# 5) C2PA embed (optional, needs c2patool)
python scripts/embed_c2pa_v21.py \
  --image results/v21_demo/watermarked.png \
  --metadata results/v21_demo/metadata.json \
  --unsigned
```
