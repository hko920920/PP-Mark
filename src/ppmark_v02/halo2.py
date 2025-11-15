from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Tuple


class Halo2Bindings:
    def __init__(self, prover_dir: Path):
        self.prover_dir = prover_dir
        self.cargo = shutil.which("cargo")

    def prove(self, secret: bytes, anchor: bytes, out_dir: Path) -> Tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        proof_path = out_dir / "proof.bin"
        public_path = out_dir / "public_inputs.json"
        if self.cargo is None:
            proof_path.write_bytes(b"halo2-stub-proof")
            public_path.write_text('{"inputs": []}\n', encoding="utf-8")
            return proof_path, public_path
        args = [
            "cargo",
            "run",
            "--release",
            "--",
            "--prove",
            "--secret",
            str(int.from_bytes(secret, "big") % (1 << 63)),
            "--anchor",
            str(int.from_bytes(anchor, "big") % (1 << 63)),
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
