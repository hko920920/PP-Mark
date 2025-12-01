# PP-Mark v0.3 — ZK-GenGuard V2 Reference

This repository contains the v0.3 implementation of PP-Mark (a.k.a. ZK-GenGuard V2).  The code focuses on the new Poseidon/RS binding, deterministic sampling, Halo2 interface, and the CPU-to-CUDA watermark embedding pipeline.  Legacy v0.2 artifacts are kept only for backward-compatibility (`src/ppmark_v02/`).

## Features
- Deterministic 1% sampling (`src/ppmark_v03/sampling.py`) and spread-spectrum embedding (`src/ppmark_v03/embedding.py`).
- Poseidon + BN254 EdDSA payload binding (`src/ppmark_v03/crypto.py`, `src/ppmark_v03/keys.py`).
- Halo2 prover/verifier packaging with stub runner (`halo2_runner.py`, `scripts/halo2_stub.py`).
- CLI for prover/verifier (`python -m ppmark_v03.cli ...`).
- Attack suite scaffold (`scripts/run_attack_suite.py`) enumerating Müller/Zhao/geometry/quality/ZKP tests.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the prover (CPU path):
```bash
python -m ppmark_v03.cli prover \
    --config config.json \
    --prompt "PP-Mark v0.3 demo" \
    --output out_demo
```

Run the verifier:
```bash
python -m ppmark_v03.cli verifier \
    --config config.json \
    --metadata out_demo/metadata.json
```

Attack suite placeholder:
```bash
python scripts/run_attack_suite.py \
    --config config.json \
    --metadata out_demo/metadata.json \
    --output attack_logs
```
Results are marked `skipped` until GPU/CUDA/Halo2 tooling is connected.

## Halo2 & CUDA Notes
- The CLI expects `config.json` to point to `tables/invcdf_gaussian.bin` and Halo2 commands.  By default it uses `scripts/halo2_stub.py`.
- To enable CUDA, install `cupy` on a GPU server (`libnvrtc.so.12` required) and invoke the prover with `DeviceConfig(backend="cuda")` inside the CLI or modify the config accordingly.

## Repo Layout
```
README.md                    # current file
pyproject.toml               # Python package metadata
scripts/                     # CLI helpers, halo2 stub, attack suite
src/ppmark_v03/              # v0.3 implementation
src/ppmark_v02/              # legacy v0.2 LiteVAE stack (reference only)
docs/architecture.md         # v0.3 architecture notes
halo2_prover/                # placeholder halo2 Rust workspace (copied)
out_demo*/, out_stub/        # sample outputs (included intentionally)
attack_logs/                 # attack suite reports
```

## Next Steps
- Implement CUDA kernels and replace the CuPy placeholder.
- Connect the real Halo2 prover/verifier and measure performance.
- Add Blind Sync + DDIM inversion for full extraction.
- Wire IPFS/Solidity hooks using `docs/ipfs_solidity_plan.md` as reference.
