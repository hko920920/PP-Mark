from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import ProviderConfig
from .crypto import generate_master_secret
from .extractor import NeuralExtractorWrapper, RobustExtractor
from .halo2 import Halo2Bindings
from .litevae import LiteVAE
from .patterns import SemanticPatternInjector
from .models.extractor_net import NeuralExtractor


@dataclass
class ProviderArtifacts:
    config_path: Path
    secret_hex: str


def create_provider_config(
    output_path: Path,
    halo2_dir: Path,
    *,
    base_image: Path | None = None,
) -> ProviderArtifacts:
    secret = generate_master_secret()
    cfg = ProviderConfig(
        master_secret_hex=secret.hex(),
        halo2_prover_dir=halo2_dir.resolve(),
        default_base_image=base_image.resolve() if base_image else None,
    )
    cfg.dump(output_path)
    return ProviderArtifacts(config_path=output_path, secret_hex=cfg.master_secret_hex)


class ProviderRuntime:
    def __init__(self, config: ProviderConfig):
        self.config = config
        litevae = LiteVAE(config.litevae)
        if config.litevae_weights and config.litevae_weights.exists():
            with config.litevae_weights.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            litevae.config.conditioning_strength = float(payload.get("conditioning_strength", litevae.config.conditioning_strength))
        self.litevae = litevae
        if config.extractor_weights and NeuralExtractor.is_available():
            neural = NeuralExtractor.from_checkpoint(
                config.extractor_weights,
                payload_bits=config.extractor.payload_bits,
            )
            self.extractor = NeuralExtractorWrapper(neural, config.extractor.payload_bits)
        else:
            self.extractor = RobustExtractor(litevae, config.extractor)
        if config.pattern_weights and config.pattern_weights.exists():
            with config.pattern_weights.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            params = payload.get("best_params", {})
            config.pattern.amplitude = float(params.get("amplitude", config.pattern.amplitude))
            config.pattern.radial_frequency = int(params.get("radial_frequency", config.pattern.radial_frequency))
        self.pattern = SemanticPatternInjector(config.pattern)
        self.halo2 = Halo2Bindings(config.halo2_prover_dir)

    @property
    def secret(self) -> bytes:
        return self.config.master_secret
