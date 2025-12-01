# PP-Mark v0.3 Architecture Notes

This document summarizes the current pp_mark_v0.3 codebase (the files under `src/ppmark_v03/`).  It replaces the legacy v0.2/v2.0 description and matches the "ZK-GenGuard V2" spec.

## Module Breakdown

| Path | Purpose |
| ---- | ------- |
| `config.py` | Defines strongly-typed dataclasses (`ImageConfig`, `ZKConfig`, etc.) and JSON load/dump helpers. |
| `crypto.py` | Poseidon-friendly hashing helpers, prompt hashing, payload binding, and deterministic sampling seeds. |
| `keys.py` | BN254 EdDSA keypair generation via `py-ecc`. |
| `payload.py` | Builds the payload binding, RS(64,32) codeword, and bitstream. |
| `rs.py` | Thin wrapper over `reedsolo` with validation helpers. |
| `sampling.py` | Deterministic 1% coordinate selection + binary trace logger for sampled observations. |
| `tables.py` | Inverse CDF lookup loader; consumes `tables/invcdf_gaussian.bin`. |
| `noise.py` | CPU reference for spread-spectrum noise synthesis (Poseidon uniforms + LUT lookup). |
| `embedding.py` | End-to-end watermark injection over the latent grid + sample trace extraction. |
| `cuda.py` | Kernel interface abstraction; currently provides CPU implementation and a CuPy-based placeholder. |
| `halo2_interface.py` | Public/witness structs shared between Python and Halo2. |
| `halo2_runner.py` | Serializes inputs and invokes an external Halo2 prover/verifier (stubbed by default). |
| `cli.py` | Typer-free CLI entrypoint (`python -m ppmark_v03 ...`) for prover/verifier flows. |

## Prover Flow (`ppmark_v03.cli`)

1. Load `config.json` via `GlobalConfig` and parse CLI arguments (prompt, seed, optional model/registry metadata).
2. Generate/resolve EdDSA keys.
3. Build payload (prompt hash → Poseidon binding → RS codeword → bitstream).
4. Deterministically select sample coordinates using the RS codeword as key material.
5. Load the inverse CDF table and obtain a `WatermarkKernel` (CPU by default, CUDA in GPU setups).
6. Embed watermark noise and collect sample trace (`SampleTrace`).
7. Serialize Halo2 public inputs + witness (`halo2_interface`), store artifacts under `out_*`.
8. Call the configured Halo2 prover command (stub by default) through `halo2_runner.py`.
9. Emit `metadata.json` containing prompt/payload hashes, halo2 file paths, and optional blockchain/IPFS placeholders.

## Verifier Flow

1. Load metadata + `config.json`.
2. Recompute deterministic sample set from the stored codeword and compare with the recorded trace (length + Merkle root).
3. Load the Halo2 proof paths from metadata and call the verifier command (stub or actual Rust binary).
4. Print status; future work will plug in Blind Sync + DDIM inversion before feeding the recovered payload bits into Halo2.

## GPU & Halo2 Notes

- `cuda.py` is structured so that GPU kernels can drop in without touching the CLI.  The CuPy implementation currently recomputes CPU noise for parity while waiting for a real CUDA kernel; the execution plan requires running on a GPU server with `libnvrtc.so.12` available.
- Halo2 integration is mediated by JSON inputs + environment variables (`HALO2_PUBLIC`, `HALO2_WITNESS`, `HALO2_PROOF`).  `scripts/halo2_stub.py` exists purely to exercise the workflow until the real Rust prover lands.

## Attack/Test Harness

`scripts/run_attack_suite.py` enumerates the required evaluations (Müller forge/erasure/re-prompt, Zhao re-generation, geometry suite, image quality, ZKP performance).  Each entry is currently marked "skipped" until the GPU pipeline and external tools are available.
