import json
from pathlib import Path

import numpy as np

from ppmark_v21.config import ProviderConfig
from ppmark_v21.crypto import generate_master_secret
from ppmark_v21.patterns import SemanticPatternInjector
from ppmark_v21.payload import bytes_to_bits
from ppmark_v21.prover import ProverService
from ppmark_v21.provider import ProviderRuntime, create_provider_config
from ppmark_v21.utils import save_image
from ppmark_v21.verifier import VerifierService


def test_prover_verifier_stub(tmp_path: Path):
    # Build a minimal provider config with stubbed halo2 (cargo absent) and default components.
    cfg_path = tmp_path / "provider.json"
    create_provider_config(output_path=cfg_path, halo2_dir=tmp_path)  # stub halo2 dir
    cfg = ProviderConfig.load(cfg_path)
    runtime = ProviderRuntime(cfg)

    # Create a simple base image.
    base_image = np.zeros((64, 64, 3), dtype=np.uint8)
    base_path = tmp_path / "base.png"
    save_image(base_path, base_image)

    prover = ProverService(runtime)
    outputs = prover.generate(
        prompt="test",
        seed="seed",
        model_id="model",
        timestamp="2025-11-18T12:10:30Z",
        output_dir=tmp_path / "out",
        base_image=base_path,
    )

    with outputs.metadata_path.open("r", encoding="utf-8") as handle:
        claimed = json.load(handle)

    verifier = VerifierService(runtime)
    result = verifier.verify(outputs.image_path, outputs.metadata_path.parent, claimed)
    assert result.is_valid  # halo2 stub returns True when cargo missing
    assert result.C_claimed == result.C_extracted

