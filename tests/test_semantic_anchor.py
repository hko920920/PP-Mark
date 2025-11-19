import numpy as np

from ppmark_v21.semantic_anchor import compute_semantic_anchor


def test_semantic_anchor_deterministic():
    h1 = compute_semantic_anchor(
        prompt="test prompt",
        seed="123",
        model_id="model-x",
        timestamp="2025-11-18T12:10:30Z",
    )
    h2 = compute_semantic_anchor(
        prompt="test prompt",
        seed="123",
        model_id="model-x",
        timestamp="2025-11-18T12:10:30Z",
    )
    assert h1 == h2


def test_semantic_anchor_changes_on_input():
    h1 = compute_semantic_anchor("p1", "1", "m", "t")
    h2 = compute_semantic_anchor("p2", "1", "m", "t")
    assert h1 != h2

