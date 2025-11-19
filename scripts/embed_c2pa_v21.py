#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed PP-Mark v2.1 metadata into a C2PA manifest (optional unsigned mode).")
    parser.add_argument("--image", type=Path, required=True, help="입력 이미지 (watermarked).")
    parser.add_argument("--metadata", type=Path, required=True, help="ppmark-v21 메타데이터 JSON (h_hex, C_hex 등).")
    parser.add_argument("--out", type=Path, required=True, help="C2PA를 삽입할 출력 이미지 경로.")
    parser.add_argument(
        "--c2patool",
        type=str,
        default="c2patool",
        help="c2patool 실행 파일 경로(기본: PATH 검색). 없으면 사이드카만 작성.",
    )
    parser.add_argument("--manifest", type=Path, help="사용자 지정 manifest JSON (없으면 자동 생성).")
    parser.add_argument("--unsigned", action="store_true", help="서명 없이 claim만 삽입.")
    parser.add_argument("--signing-cert", type=Path, help="서명 인증서(.pem).")
    parser.add_argument("--signing-key", type=Path, help="서명 키(.pem).")
    return parser.parse_args()


def build_manifest(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "claim_generator": "pp-mark-v2.1",
        "assertions": [
            {
                "label": "ppmark-v21-metadata",
                "data": {
                    "type": "application/json",
                    "json": metadata,
                },
            }
        ],
        "features": {
            "watermark": {
                "l1": metadata.get("C_hex"),
                "l2": metadata.get("C_hex"),
            }
        },
    }


def ensure_output_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_c2patool(
    tool: str,
    image: Path,
    manifest: Path,
    out_path: Path,
    *,
    unsigned: bool,
    signing_cert: Path | None,
    signing_key: Path | None,
) -> None:
    cmd = [tool, str(image), "--manifest", str(manifest), "--out", str(out_path)]
    if unsigned:
        cmd.append("--unsigned")
    if signing_cert and signing_key:
        cmd.extend(["--signcert", str(signing_cert), "--signkey", str(signing_key)])
    print(f"[c2pa] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")
    if not args.metadata.exists():
        raise SystemExit(f"metadata not found: {args.metadata}")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    manifest_data = build_manifest(metadata)

    ensure_output_dirs(args.out)
    claim_sidecar = args.out.with_suffix(args.out.suffix + ".c2pa.json")
    claim_sidecar.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    tool_path = shutil.which(args.c2patool) if args.c2patool == "c2patool" else args.c2patool

    if tool_path is None:
        print("[c2pa] c2patool not found; copying image and writing sidecar JSON only.", file=sys.stderr)
        shutil.copy2(args.image, args.out)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = args.manifest or Path(tmpdir) / "ppmark_manifest.json"
        if not args.manifest:
            manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        run_c2patool(
            tool_path,
            args.image,
            manifest_path,
            args.out,
            unsigned=args.unsigned,
            signing_cert=args.signing_cert,
            signing_key=args.signing_key,
        )
    print(f"[c2pa] Embedded manifest into {args.out}")


if __name__ == "__main__":
    main()
