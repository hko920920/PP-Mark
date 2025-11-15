# PP-Mark v0.2 — LiteVAE Dual-Channel Prototype

`pp_mark_v0.2` is a self-contained upgrade of the original PP-Mark prototype.  It mirrors the CLI-oriented structure of `pp_mark/` but implements the ICML-ready **PP‑Mark v2.0** architecture:

1. **Stage 1 — Provider setup**  
   Generate the master secret `k`, Halo2 proving/verification keys, LiteVAE checkpoints, PatternGen parameters, and the crop-robust extractor `W`.  The helper script `scripts/provider_setup.py` writes everything into `artifacts/provider_setup.json`.
2. **Stage 2 — Image + proof generation (prover side)**  
   `scripts/run_prover.py` produces a watermarked image, Poseidon payload, Halo2 proof, and structured metadata for a given prompt or base image.  LiteVAE with DWT sub‑bands embeds payload bits in the spatial channel while a Robin-style `PatternGen(C)` perturbs the diffusion latent mid‑trajectory.
3. **Stage 3 — Public verification**  
   `scripts/run_verifier.py` recomputes pHash / payload via the extractor and runs Halo2 verification.  When attacked images break either channel, the verifier still surfaces mismatches and refuses to validate.

The repository keeps the same ergonomic affordances as v1:

- `assets/` provides canonical demo images (COCO street market sample).  
- `results/` collects generated outputs, proofs, and JSON metadata.  
- `halo2_prover/` is copied verbatim from the v1 workspace, so existing Rust tooling and parameter files continue to work.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Optional (training extras)
pip install -e .[train]
```

### Typer CLI (recommended)

```bash
# Stage 1 — provider setup
python -m ppmark_v02 setup \
    --out artifacts/provider_setup.json \
    --halo2-dir halo2_prover \
    --base-image assets/processed/coco/coco_val_street_market.png

# Stage 2 — prover
python -m ppmark_v02 prover \
    --config artifacts/provider_setup.json \
    --prompt "PP-Mark LiteVAE demo" \
    --output-dir results/demo

# Stage 3 — verifier
python -m ppmark_v02 verify \
    --config artifacts/provider_setup.json \
    --image results/demo/watermarked.png \
    --metadata results/demo/metadata.json
```

### One-shot demo runner

```bash
python scripts/run_full_demo.py \
    --prompt "PP-Mark LiteVAE demo" \
    --output-dir results/demo_run
```

`scripts/run_full_demo.py` automatically generates the provider config (unless it already exists), runs the prover, and immediately verifies the resulting artifact.  Use `--skip-verify` to stop after Stage 2.  You can also install the package in editable mode (`pip install -e .`) thanks to `pyproject.toml`, which exposes the `ppmark-v02` console script that wraps the same Typer CLI.

## Model Training / Tuning

Stage 1 requires preparing three assets: extractor `W`, LiteVAE decoder strength, and the semantic PatternGen injector.  The repo now ships self-contained scripts that implement the ICML spec’s training flow:

```bash
# 1) Extractor (PyTorch CNN trained on augmentations)
python scripts/train_extractor.py \
    --config artifacts/provider_setup.json \
    --data-root assets/processed/coco \
    --output runs/extractor/latest.pt

# 2) LiteVAE conditioning strength sweep
python scripts/tune_litevae.py \
    --config artifacts/provider_setup.json \
    --data-root assets/processed/coco \
    --output runs/litevae_strength.json

# 3) PatternGen amplitude/frequency selection
python scripts/train_pattern.py \
    --config artifacts/provider_setup.json \
    --data-root assets/processed/coco \
    --output runs/pattern_selection.json
```

After training, point `ProviderConfig` to the generated artifacts (`extractor_weights`, `litevae_weights`, `pattern_weights`) so that `ProviderRuntime` automatically loads them when running the prover or verifier.  All scripts rely on the shared dataset/augmentation pipeline under `src/ppmark_v02/training/`.

### Attack Harness

To reproduce the ICML attack-matrix sanity checks on the new LiteVAE pipeline, call:

```bash
python scripts/run_attack_matrix.py \
    --config artifacts/provider_setup.json \
    --image results/demo/watermarked.png \
    --metadata results/demo/metadata.json
```

The script applies JPEG compression, Gaussian blur, and 90% center crop by default.  Additional transforms (down/up sampling, flips, Poisson noise, perspective) can be added by editing `attack_suite()`.  Rotations/elastic warps—which are known to be failure modes—will be evaluated later under a dedicated “stress” profile once the base extractor is tuned.

See `docs/architecture.md` for the full PP-Mark v2.0 breakdown.  The Python modules inside `src/ppmark_v02/` are intentionally lightweight (NumPy/PyWavelets powered) so the logic is inspectable and runnable on CPU while preserving the dual-channel design decisions from the design brief.
