# PP-Mark v2.0 Architecture Notes

This document mirrors the verbal specification and tracks how the Python modules fulfill each requirement.  Operationally, the Typer CLI (`python -m ppmark_v02 ...` or the `ppmark-v02` console script) mirrors the three stages, while `scripts/run_full_demo.py` chains them for a quick sanity run.

## Stage 1 — Provider Setup

- `ppmark_v02.crypto` exposes `generate_master_secret()` and `poseidon_payload(secret, phash)` so the master key `k`, Poseidon payload `C`, and Halo2 bindings are consistent.  
- `scripts/provider_setup.py` writes a JSON blob with:
  - `master_secret_hex`
  - `halo2_prover_dir` (the copied Rust circuit in `halo2_prover/`)
  - LiteVAE config snapshot
  - Extractor/training metadata (conditioning strength, crop tiling strategy, augmentation seeds)
- LiteVAE is implemented via DWT multi-resolution decomposition (`ppmark_v02.litevae`).  Conditioning vectors derived from payload bits are inserted into the finest detail band, mirroring the “channel 2” story.  
- The extractor `ppmark_v02.extractor.RobustExtractor` uses the same tiling/permutation to recover payload bits even when the image is center-cropped to 90%.  When the PyTorch-based extractor is trained via `scripts/train_extractor.py`, `ProviderRuntime` switches to the neural checkpoint transparently.
- `scripts/tune_litevae.py` and `scripts/train_pattern.py` perform exhaustive sweeps over LiteVAE conditioning strength and PatternGen amplitudes/frequencies, respectively.  Their outputs (`litevae_weights`, `pattern_weights`) are JSON blobs that the runtime reads to override default hyperparameters without code changes.

## Stage 2 — Dual-Channel Embedding

`ppmark_v02.prover.ProverService` executes the numbered steps from the spec:

1. **Anchor** — `ppmark_v02.hashing.compute_phash()` computes a 256‑bit perceptual hash from the base image (generated via Diffusers when available, otherwise sampling from `assets/processed/coco`).
2. **Payload** — `C = poseidon_payload(k, h)` uses the exact `Poseidon(k || h)` interface.  The helper transparently falls back to SHA3 if Poseidon bindings are not installed, but the Halo2 circuit still expects Poseidon so both hashes are recorded in metadata.
3. **Channel 1** — `ppmark_v02.patterns.SemanticPatternInjector` produces `PatternGen(C)` (structured radial/sinusoidal frequency masks) and injects it into the latent `x_t` replica.  The simulator keeps `t_injection` configurable.
4. **Channel 2** — LiteVAE decodes `z_T` conditioned on the payload vector through `D'_Lite`.  Because LiteVAE already stores multi-resolution detail subbands, the conditioner simply adds payload-aligned perturbations to HH coefficients before `pywt.waverec2`.
5. **Proof** — `ppmark_v02.halo2.Halo2Bindings` shells out to `halo2_prover` to run `cargo run -- --prove ...` with consistent inputs, persisting `proof.bin` alongside the raw public inputs.

`scripts/run_prover.py` glues everything together: prompts, noise seeds, crop augmentation for extractor audit, and JSON outputs containing `h_hex`, `C_hex`, and Halo2 file references.

## Stage 3 — Verification

`ppmark_v02.verifier.VerifierService` performs:

1. Payload extraction via the `RobustExtractor`.  It tries the full image, a 90% center crop, and resized variants before giving up.  
2. Metadata comparison to guard against tampering before the expensive ZKP step.  
3. Halo2 verification through the same Rust binary (`cargo run -- --verify path/to/dir`).  The CLI caches the verification key path from the Stage 1 config.

On success the verifier returns a structured report (JSON + CLI output) summarizing the per-channel status and the Halo2 verdict.  On failure it reports which check triggered (pHash mismatch, extractor mismatch, proof failure).

`scripts/run_attack_matrix.py` mirrors the classic PP-Mark attack harness: it replays JPEG/Gaussian/crop perturbations against a generated image and reports whether the newly trained extractor can still recover the payload bits, keeping the Müller/Diffusion evaluation workflow grounded even before GPU-scale experiments.

## Pending High-Intensity Evaluation

Some metrics described in the PP-Mark v2.0 brief require GPU-grade workloads or external toolchains.  They are tracked (but not executed yet) as follows:

- **Geometric/Noise Attacks**: `scripts/run_attack_matrix.py` currently covers JPEG compression, Gaussian blur, and 90% center crop.  Additional transforms (down/up sampling, random flips, Poisson noise, perspective warp) can be plugged into `attack_suite()` as needed.
- **Rotation/Elastic Warp (High-Criticality)**: Expected to be failure cases—plan to group them under a “stress” profile (small ±15° rotations, elastic warp, affine shear) and log them separately once the base extractor is validated.
- **Müller Imprint Forgeries (3 settings)**: Requires the `external/semantic-forgery` repository plus an H100/A100 server.  The plan is to reuse the v1 launcher (`scripts/run_mueller_attack.py`) once GPU access is available, logging cover-prompt choices and verifying recovered payload/ZKP status with the v0.2 verifier.
- **Zhao Diffusion Re-generation Attack**: Requires diffusers + WatermarkAttacker (GPU).  To be executed through `scripts/run_diffusion_regen.py` with varying `noise-step`, then recorded through the same attack-matrix JSON schema.
- **Image Quality Metrics**: LPIPS/SSIM computation hooks will be added next to the prover outputs once `lpips` and `skimage` dependencies are installed; the README now notes that this measurement is pending until scientific Python deps are installed.
- **Performance Numbers**:
  - *Halo2 Prover/Verifier*: Time measurement will be captured via the Rust CLI (`cargo run -- --prove/--verify …`) once the toolchain is available; logs will be stored beside `proof.bin`.
  - *Extractor / Embed Latency*: Extractor wrappers already expose Python methods, so timing decorators will be added once GPUs (for CNN inference) are accessible.

All items above remain TODOs solely because the current environment blocks GPU/large-dependency execution; the code paths and documentation have placeholders that point to the exact scripts/commands to run once those resources are provisioned.
