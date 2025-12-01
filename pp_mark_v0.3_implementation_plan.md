# ZK-GenGuard V2 (pp_mark_v0.3) Implementation Plan

This document captures the requirements and progress for the pp_mark_v0.3 implementation. All metrics assume 1080×1080 SDXL latent grids unless specified.

---

## 1. Objectives & Success Targets
- Watermark embed → extract → Halo2 proof chain completes within 1.5s on RTX 4090 (GPU path).
- Pure lookup/addition-friendly circuits (Poseidon binding, spread spectrum) without FFT-heavy ops.
- Optimistic verification: metadata fast path with fallback to blockchain/IPFS when absent.
- Statistical security via deterministic 0.5–2% sampling (default 1%).

## 2. System Overview
1. **Setup**: register EdDSA(BN254) keys + Halo2 VK on chain/IPFS (future work).
2. **Payload**: prompt/seed/secret → Poseidon binding → RS(64,32) encoding.
3. **Embedding**: Poseidon CSPRNG → inverse CDF LUT → spread spectrum noise injected into SDXL latent grid.
4. **Proving**: sample ~11k coordinates, log uniforms/gaussian/combined values, run Halo2 circuit.
5. **Verification**: metadata path verifies immediately; recovery path replays extraction + halo2 verification.

## 3. Configuration & Tables
- `config.py` provides dataclasses with JSON load/dump.
- `tables/invcdf_gaussian.bin` stores the inverse CDF lookup; generated via `scripts/generate_inverse_cdf.py` (2^16 points).

## 4. Modules (under `src/ppmark_v03`)
- `crypto.py`, `keys.py`, `payload.py`, `rs.py`: Poseidon binding, BN254 keygen, RS encoding, helpers.
- `sampling.py`, `tables.py`, `noise.py`, `embedding.py`, `cuda.py`: deterministic sampling, LUT lookups, CPU/CUDA kernels.
- `halo2_interface.py`, `halo2_runner.py`: JSON packaging for Halo2, external command hooks, stub fallback.
- `cli.py`: prover/verifier entry points; metadata includes blockchain/IPFS placeholders.

## 5. Scripts
- `scripts/halo2_stub.py`: placeholder prover/verifier.
- `scripts/run_attack_suite.py`: enumerates Müller/Zhao/geometry/quality/ZKP tests (marked skipped until GPU/external tooling is available).

## 6. Roadmap
1. **CPU reference**: ✅ config, Poseidon, RS, sampling, noise, CLI, halo2 stub.
2. **GPU integration**: CuPy placeholder ready; real CUDA kernels pending GPU server.
3. **Halo2 Rust prover**: JSON packaging done; connect actual circuit + measure latency next.
4. **Blind Sync + DDIM extraction**: not implemented yet.
5. **Blockchain/IPFS hooks**: metadata placeholders ready; implement upload + registry update later.

## 7. Risks & Testing
- **CUDA**: current machine lacks `libnvrtc.so.12`; must run GPU code on a server/Jupyter environment.
- **Attack harness**: `scripts/run_attack_suite.py` logs tests as `skipped` until GPU/external tools are wired.
- **Deterministic sampling leakage**: only expose hashed commitments; indices stay private until proof.
- **DDIM inversion failure**: to be addressed alongside sync module build-out.
- **Testing**: once GPU/Halo2 available, rerun Müller/Zhao/geometry suites and Halo2 benchmarks.

---

This plan will be updated as new modules (CUDA kernels, Halo2 circuit bindings, sync/extraction) land.
