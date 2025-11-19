from __future__ import annotations

import hashlib
from typing import Optional


def compute_semantic_anchor(
    prompt: str,
    seed: str | int,
    model_id: str,
    timestamp: str,
    parent_manifest_hash: Optional[str] = None,
) -> bytes:
    """
    Build a deterministic semantic anchor from generation metadata.

    This replaces pHash-based anchors so the value remains stable regardless of
    downstream watermarking perturbations.
    """
    parts = [
        prompt.strip(),
        str(seed),
        model_id.strip(),
        timestamp.strip(),
    ]
    if parent_manifest_hash:
        parts.append(parent_manifest_hash.strip())
    joined = "|".join(parts).encode("utf-8")
    return hashlib.sha256(joined).digest()
