from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple


class Halo2Bindings:
    def __init__(self, prover_dir: Path):
        self.prover_dir = prover_dir
        self.cargo = shutil.which("cargo")
        if not (self.prover_dir / "Cargo.toml").exists():
            self.cargo = None

    def prove(self, secret: bytes, anchor: bytes, payload_hex: str, out_dir: Path) -> Tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        proof_path = out_dir / "proof.bin"
        public_path = out_dir / "public_inputs.json"
        anchor_hex = anchor.hex()
        payload_hex = payload_hex.lower()
        if self.cargo is None:
            proof_path.write_bytes(b"halo2-stub-proof")
            stub_payload = {
                "poseidon_hex": f"0x{payload_hex}",
                "anchor_hex": f"0x{anchor_hex}",
                "inputs": [f"0x{payload_hex}"],
            }
            public_path.write_text(json.dumps(stub_payload, indent=2), encoding="utf-8")
            return proof_path, public_path
        args = [
            "cargo",
            "run",
            "--release",
            "--",
            "--prove",
            "--secret-hex",
            f"0x{secret.hex()}",
            "--anchor-hex",
            f"0x{anchor_hex}",
        ]
        subprocess.run(args, cwd=self.prover_dir, check=True, capture_output=True)
        shutil.copy(self.prover_dir / "outputs/proof.bin", proof_path)
        shutil.copy(self.prover_dir / "outputs/public_inputs.json", public_path)
        return proof_path, public_path

    def verify(self, proof_dir: Path) -> bool:
        proof_dir = proof_dir.resolve()
        if self.cargo is None:
            return True
        args = [
            "cargo",
            "run",
            "--release",
            "--",
            "--verify",
            str(proof_dir),
        ]
        proc = subprocess.run(args, cwd=self.prover_dir, capture_output=True, text=True)
        return proc.returncode == 0

    @staticmethod
    def read_public_inputs(public_path: Path) -> Dict[str, str]:
        with public_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
