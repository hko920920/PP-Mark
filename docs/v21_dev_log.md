# PP-Mark v2.1 Development Log

This log tracks every change performed during the Codex session (2025-01-07). It explains **what was touched, why it was necessary, what references/observations motivated the change, and how it was validated**. Future contributors can resume work seamlessly from here.

---

## 1. Repository Re-alignment to v2.1
- **Why**: The root `README.md`, `pyproject.toml`, and `requirements.txt` described the legacy v0.2 workflow, but the design brief (cf. `docs/v21_summary.md`) required semantic anchors + dual watermark defense.
- **What**:
  - Rewrote `README.md` to describe v2.1 CLI, seed guidance, attack scripts, and C2PA helper.
  - Updated `pyproject.toml` metadata (`name=ppmark-v2-1`, `version=2.1.0`) and exposed a `ppmark-v21` console entrypoint.
  - Added `poseidon-py` to `requirements.txt` so the cryptographic layer matches Poseidon/Halo2 requirements.
  - Added `scripts/embed_c2pa_v21.py` to embed metadata via `c2patool` (or generate a sidecar if unavailable).
- **Validation**: Manual inspection + CLI smoke tests (`ppmark-v21 --help`). No automated tests depended on the README or packaging, so none required.

---

## 2. Stable Signature (L1) + LiteVAE Integration
- **Why**: Prover previously injected payload by a single LiteVAE encode/decode pass, so extractor mismatches were common. Spec requires Stable Signature style decoder fine-tune (iterative embedding).
- **What**:
  - Added `src/ppmark_v21/stable_signature.py` implementing iterative LiteVAE conditioning with extractor feedback.
  - `ProviderRuntime` now exposes `signature_encoder`; `ProverService.generate()` calls it with pattern post-processing and pre-extraction removal hooks.
  - Introduced deterministic coefficient index selection (`idx.sort()`) and rounding to reduce floating drift.
- **Challenges**:
  - Initial attempts failed because pattern injection happened before LiteVAE decode; the extractor saw only blended noise. Invented `postprocess` (apply pattern after decoding) and `pre_extract` (remove pattern before extraction).
  - Extractor failed even with deterministic conditioning because DWT detail bands collapsed to only two values (0.498/1.0). Added dynamic thresholding: if the sampled coefficients are binary, classify via distance to min/max rather than median.
- **Validation**:
  - Created `tests/test_prover_verifier_stub.py` to run the full setup/prover/verifier loop (cargo absent, stub proof). Test now passes and ensures extractor recovers payload bits exactly.
  - Manual debug scripts (`tmp_debug*`, now deleted) confirmed `extraction.bits` == `payload_bits` after adjustments.

---

## 3. ROBIN Pattern (L2) Adjustments
- **Why**: Original pixel-additive pattern drastically changed pixel ranges, making L1 extraction impossible. Need subtle blending + invertible removal.
- **What**:
  - `SemanticPatternInjector.apply()` now blends the image with a [0,1] sinusoid via convex combination (amplitude as alpha).
  - Added `remove()` method to invert the blend; used before extraction and inside the signature encoder.
  - Added multi-scale frequency correlation + heuristic fallback remain as in spec; only injection pipeline changed.
- **Validation**: Verified in tests that pattern removal before extraction enables convergence (see Section 2). No dedicated L2 detection tests yet—future work.

---

## 4. Mid-Trajectory ROBIN Hook
- **Why**: Section 3 only blended patterns at the final image, whereas the spec requires a ROBIN-style mid-trajectory injection (`t_injection`, hiding prompt guidance). We needed an explicit injector that simulates diffusion forward/backward passes so L2 metadata becomes meaningful.
- **What**:
  - Added `src/ppmark_v21/robin.py` providing `MidTrajectoryInjector` + `RobinTrace`. It simulates a DDIM-like forward blur/noise phase, applies the existing pattern injector on that latent, and blends the guided result back into the base image.
  - `ProviderRuntime` now exposes `runtime.robin`; the prover first runs `robin.inject()` (records trace), then Stable Signature operates on that output. Pattern removal/extraction flows now call `runtime.robin.prepare_for_extraction()` and `runtime.robin.extract_signal*()`.
  - Metadata includes `l2_trace` (t_injection, guidance_scale, hiding_prompt, payload checksum) so verifiers know which secret path was used.
- **Challenges**:
  - Extractor initially failed because LiteVAE detail bands became binary constants after strong conditioning; the fix was deterministic coefficient selection + distance-based voting when the sample set collapses to two values.
  - Need to ensure the simulated diffusion is invertible enough for CPU tests; forward step uses blur+noise, reverse step blends with configurable `guidance_scale`.
- **Validation**:
  - Updated prover/verifier stub test covers the new pipeline (ROBIN inject → Stable Signature → verification). `pytest tests/test_prover_verifier_stub.py -q` now exercises both L1 and L2.
  - Manual debugging scripts (removed) showed `extractor.bits == payload_bits` after pattern removal and ROBIN injection.

---

## 5. Halo2 + Poseidon Binding (L3)
- **Why**: Earlier code truncated secret/anchor to 63-bit integers and lacked anchor binding in public inputs, contradicting the Poseidon/Halo2 spec.
- **What**:
  - `src/ppmark_v21/crypto.py` and `src/ppmark_v02/crypto.py` now call `poseidon_hash_many([secret_int, anchor_int])` instead of concatenating bytes and falling back to SHA3.
  - `src/ppmark_v21/halo2.py` and the Rust circuit `halo2_prover/src/main.rs` accept hex inputs, write both `poseidon_hex` and `anchor_hex` to `public_inputs.json`, and include a stub when cargo is missing.
  - `VerifierService` checks for missing proofs/public inputs, compares anchors/payloads with the declared metadata before running Halo2 verification, and fails fast on mismatches.
  - `src/ppmark_v02/prover.py` updated to pass `payload_hex` to Halo2 so v0.2 remains consistent.
- **Challenges**:
  - PyPoseidon API only exposes `poseidon_hash_many`; initial call signature (`poseidon_hash(field_elements)`) raised `TypeError: missing argument y`. Adjusted accordingly.
  - Local test env lacked `Cargo.toml` for temp dirs; detection added so `self.cargo` is `None` unless the prover dir actually contains the Rust workspace, preventing `subprocess.CalledProcessError: could not find Cargo.toml`.
- **Validation**:
  - `tests/test_prover_verifier_stub.py` ensures signature, L1 extraction, and `public_inputs.json` consistency are enforced even with stubbed Halo2.
  - Manual `poseidon_payload` vs. metadata assertions performed during debugging.

---

## 6. CLI + Docs + Tests Additions
- Added `tests/test_semantic_anchor.py` for deterministic SHA256 anchor generation (mirrors design doc).
- Created `docs/v21_summary.md`, `docs/v21_attack_plan.md`, `docs/v21_attack_steps.md`, `docs/seed_guidance_v21.md`, and sample metadata files to match the ICML brief.
- Extended Typer CLI (`ppmark_v21/cli.py`) to expose setup/prover/verify commands referencing the new config path (`artifacts/provider_setup_v21.json`).
- Added `scripts/run_full_demo_v21.py`, `scripts/run_attack_matrix_v21.py`, `scripts/run_regen_attack_v21.py`, `scripts/run_regen_surrogate_v21.py`, and `scripts/run_mueller_attack.py` to mirror the doc’s recommended evaluation flows. GPU-dependent scripts still require external repos; these are annotated in README/docstrings.
- Metadata signing: refactored signer 로직을 모듈화(`signing.py`)하여 기본값을 `ecdsa-p256` 전자서명으로 전환했으며, `signature_scheme` 필드를 통해 HMAC-SHA256 등 다른 스킴으로도 교체할 수 있도록 구성했다. 메타데이터에는 공개키(`signature_pubkey_hex`)와 스킴 정보가 함께 들어간다.
- Added `scripts/eval_image_quality.py` (PSNR/SSIM + optional LPIPS) so that base vs. watermarked 이미지 품질을 빠르게 비교할 수 있으며, `scikit-image` 의존성이 추가되었다.
- All tests (`pytest -q`) pass: 5 total (2 new, 3 existing).

---

## 7. Training Scripts (LiteVAE / Pattern / Extractor)
- **Why**: `scripts/train_extractor.py`, `scripts/tune_litevae.py`, and `scripts/train_pattern.py` still imported `ppmark_v02.*`, so Stage 1 assets would diverge from the v2.1 runtime.
- **What**:
  - Added `src/ppmark_v21/training/` (configs, datasets, extractor_trainer) mirroring the v0.2 utilities but driving Poseidon payloads via synthetic semantic anchors derived from dataset paths.
  - Training dataset samples now include their source paths so anchors can reference deterministic metadata (prompt=filename, parent hash=path SHA256, seed=index).
  - Updated all three scripts to import `ppmark_v21` modules; tooling remains CLI-compatible (`--config`, `--data-root`, etc.) but now produces artifacts aligned with the v2.1 provider runtime.
- **Validation**: `pytest -q` still passes (training modules are only imported when scripts run with the `train` extra installed). Running `PYTHONPATH=src python scripts/train_extractor.py --help` without PyTorch now raises a clear RuntimeError instructing the user to install `pip install -e .[train]`, confirming the dependency message works.

---

## 8. Outstanding Items / Next Steps
1. **ROBIN diffusion-level validation** *(GPU/Jupyter 서버 필요)*: 현재 인젝터는 CPU 시뮬레이션일 뿐이므로, diffusers/DDIM/SDXL 파이프라인에 직접 `t_injection` + hiding prompt를 삽입하는 버전은 GPU 자원이 있는 주피터/서버 환경에서 진행해야 함. 해당 환경이 마련되면 이 항목부터 재착수.
2. **Halo2 Poseidon2 parameters**: 현재 Halo2 회로는 기본 Poseidon(P128Pow5T3)을 사용하지만, Pow5 칩이 Poseidon2의 내부/외부 선형 계층 구조를 지원하지 않아 “상수 교체”만으로는 전환이 불가능합니다. HorizenLabs에서 Pallas/Vesta 상수(MAT_DIAG3_M_1, MAT_INTERNAL3, RC3 등)는 확보했지만, Poseidon2 칩 자체를 새로 구현해야 하는 작업이라 추후 전용 칩을 작성할 수 있는 환경에서 재진행이 필요합니다. 참고로 현 Poseidon/Pow5 기반 ZKP 성능은 프로버 3~4초, 베리파이어 1~2초 수준으로 보고되었습니다.
3. **Metadata signing key management** *(단계적 계획)*: 현재 기본 스킴은 `ecdsa-p256`(공개키 메타데이터 포함)이며, 필요 시 `signature_scheme`을 바꿔 HMAC-SHA256 등의 대칭 스킴으로도 전환할 수 있다. 향후 C2PA 연동이 필요할 경우 동일한 인터페이스에 ECDSA 인증서/Ed25519 등을 추가할 계획.
4. **Attack automation**: GPU scripts are placeholders; run them on a machine with Diffusers/semantic-forgery repos and log results under `results/`.

---

## Summary for Future Sessions
- To run the full pipeline locally:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  pip install -e .[train]  # optional
  ppmark-v21 setup --out artifacts/provider_setup_v21.json
  ppmark-v21 prover --config artifacts/provider_setup_v21.json --prompt ... --seed <256-bit hex> ...
  ppmark-v21 verify --config artifacts/provider_setup_v21.json --image ... --metadata ...
  pytest -q
  ```
- Halo2 proving requires a real Rust toolchain and `halo2_prover/` with `Cargo.toml`. Without it, stub proofs and public inputs are generated automatically.
- All new Python files include explicit comments/docstrings; this log explains the “why” in prose. Please append new entries to this log when adding features or fixing bugs.
