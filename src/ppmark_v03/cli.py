"""PP-Mark v0.3 CLI."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .config import GlobalConfig
from .cuda import DeviceConfig, get_watermark_kernel
from .halo2_interface import Halo2Package, PublicInputs, Witness, build_sample_commitment
from .halo2_runner import Halo2Paths, prepare_prover_package, run_halo2_prover, verify_halo2_proof
from .keys import KeyPair, derive_public_point, generate_keypair
from .payload import build_payload
from .rs import ReedSolomonCodec
from .sampling import SampleSet, SampleTrace, deterministic_sample
from .tables import InverseCDFTable


def _load_secret(secret_hex: str | None, secret_file: Path | None) -> KeyPair:
    if secret_hex:
        value = int(secret_hex, 16)
        public = derive_public_point(value)
        return KeyPair(secret=value, public=public)
    if secret_file and secret_file.exists():
        value = int(secret_file.read_text().strip(), 16)
        public = derive_public_point(value)
        return KeyPair(secret=value, public=public)
    return generate_keypair()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve(base: Path, target: str) -> Path:
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def run_prover(args: argparse.Namespace) -> None:
    cfg = GlobalConfig.load(Path(args.config))
    output_dir = Path(args.output).resolve()
    _ensure_dir(output_dir)

    codec = ReedSolomonCodec(n=cfg.watermark.rs_n, k=cfg.watermark.rs_k)
    keypair = _load_secret(args.secret_hex, Path(args.secret_file) if args.secret_file else None)
    seed = secrets.token_bytes(32) if args.seed_hex is None else bytes.fromhex(args.seed_hex)

    payload = build_payload(prompt=args.prompt, seed=seed, secret=keypair.secret_bytes(), codec=codec)
    sample_set = deterministic_sample(
        total_pixels=cfg.image.total_pixels,
        width=cfg.image.width,
        height=cfg.image.height,
        key_material=payload.codeword,
        count=cfg.image.sample_count(),
    )

    inverse_cdf = InverseCDFTable.from_file(cfg.tables.inverse_cdf_path)
    kernel = get_watermark_kernel(DeviceConfig(), cfg, inverse_cdf)
    embedding = kernel.embed(binding=payload.binding, bit_sequence=payload.bits, sample_set=sample_set)

    latent_path = output_dir / "latent_noise.npy"
    np.save(latent_path, embedding.latent_noise)
    trace_path = output_dir / "sample_trace.bin"
    embedding.sample_trace.save_binary(trace_path)

    sample_root, _ = build_sample_commitment(sample_set)
    key_path = output_dir / "key_info.json"
    key_info = {
        "secret_hex": keypair.secret_bytes().hex(),
        "public_x": hex(keypair.public[0]),
        "public_y": hex(keypair.public[1]),
    }
    key_path.write_text(json.dumps(key_info, indent=2), encoding="utf-8")

    public_inputs = PublicInputs(prompt_hash=payload.prompt_hash, binding=payload.binding, sample_merkle_root=sample_root)
    witness = Witness(
        secret_key=keypair.secret_bytes(),
        seed=seed,
        codeword=payload.codeword,
        sample_observations=list(embedding.sample_trace.entries),
    )
    halo2_dir = output_dir / "halo2"
    package = Halo2Package(public_inputs=public_inputs, witness=witness, proof_path=halo2_dir / "proof.bin")
    halo2_paths = prepare_prover_package(package, sample_set, halo2_dir)
    run_halo2_prover(halo2_paths, cfg.zk)

    metadata = {
        "prompt": args.prompt,
        "prompt_hash": hex(payload.prompt_hash),
        "binding": hex(payload.binding),
        "seed_hex": seed.hex(),
        "message_hex": payload.message_bytes.hex(),
        "codeword_hex": payload.codeword.hex(),
        "latent_noise": str(latent_path),
        "sample_trace": str(trace_path),
        "sample_count": len(sample_set.indices),
        "sample_root": sample_root.hex(),
        "key_info": str(key_path),
        "halo2": {
            "public": str(halo2_paths.public),
            "witness": str(halo2_paths.witness),
            "proof": str(halo2_paths.proof),
        },
    }
    if args.model_id or args.registry_address or args.vk_cid or args.proof_cid:
        metadata["blockchain"] = {
            "model_id": args.model_id,
            "registry_address": args.registry_address,
            "vk_cid": args.vk_cid,
            "proof_cid": args.proof_cid,
        }

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[prover] wrote payload metadata to {meta_path}")


def run_verifier(args: argparse.Namespace) -> None:
    cfg = GlobalConfig.load(Path(args.config))
    meta_path = Path(args.metadata)
    metadata: Dict[str, Any]
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    trace_path = _resolve(meta_path.parent, metadata["sample_trace"])
    sample_trace = SampleTrace.load_binary(trace_path)
    codec = ReedSolomonCodec(n=cfg.watermark.rs_n, k=cfg.watermark.rs_k)
    codeword = bytes.fromhex(metadata["codeword_hex"])
    sample_set = deterministic_sample(
        total_pixels=cfg.image.total_pixels,
        width=cfg.image.width,
        height=cfg.image.height,
        key_material=codeword,
        count=cfg.image.sample_count(),
    )
    if len(sample_trace.entries) != len(sample_set.indices):
        raise RuntimeError("Sample trace length mismatch")
    sample_root, _ = build_sample_commitment(sample_set)
    if sample_root.hex() != metadata["sample_root"]:
        raise RuntimeError("Sample root mismatch")

    halo2_meta = metadata.get("halo2")
    if not halo2_meta:
        raise RuntimeError("Halo2 metadata missing")
    halo2_paths = Halo2Paths(
        public=_resolve(meta_path.parent, halo2_meta["public"]),
        witness=_resolve(meta_path.parent, halo2_meta["witness"]),
        proof=_resolve(meta_path.parent, halo2_meta["proof"]),
    )
    verify_halo2_proof(halo2_paths, cfg.zk)
    print("[verifier] sample trace validated; halo2 verification completed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PP-Mark v0.3 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    prover = sub.add_parser("prover", help="Run prover pipeline")
    prover.add_argument("--config", required=True)
    prover.add_argument("--prompt", required=True)
    prover.add_argument("--output", required=True)
    prover.add_argument("--seed-hex")
    prover.add_argument("--secret-hex")
    prover.add_argument("--secret-file")
    prover.add_argument("--model-id")
    prover.add_argument("--registry-address")
    prover.add_argument("--vk-cid")
    prover.add_argument("--proof-cid")
    prover.set_defaults(func=run_prover)

    verifier = sub.add_parser("verifier", help="Run verifier pipeline")
    verifier.add_argument("--config", required=True)
    verifier.add_argument("--metadata", required=True)
    verifier.set_defaults(func=run_verifier)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
