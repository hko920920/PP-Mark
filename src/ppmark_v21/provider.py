from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import ProviderConfig
from .crypto import (
    generate_master_secret,
    generate_signing_key,
    generate_ecdsa_keypair,
)
from .extractor import NeuralExtractorWrapper, RobustExtractor
from .halo2 import Halo2Bindings
from .litevae import LiteVAE
from .patterns import SemanticPatternInjector
from .models.extractor_net import NeuralExtractor
from .stable_signature import StableSignatureEncoder
from .robin import MidTrajectoryInjector
from .signing import create_signer, Signer


@dataclass
class ProviderArtifacts:
    config_path: Path
    secret_hex: str


def create_provider_config(
    output_path: Path,
    halo2_dir: Path,
    *,
    base_image: Path | None = None,
    signing_key: bytes | None = None,
    signing_pubkey: bytes | None = None,
    signature_scheme: str = "ecdsa-p256",
) -> ProviderArtifacts:
    secret = generate_master_secret()
    scheme = signature_scheme.lower()
    if scheme in {"hmac", "hmac-sha256"}:
        signing_key = signing_key or generate_signing_key()
        signing_pubkey = None
    elif scheme in {"ecdsa", "ecdsa-p256"}:
        if signing_key is None or signing_pubkey is None:
            priv, pub = generate_ecdsa_keypair()
            signing_key = priv
            signing_pubkey = pub
    else:
        raise ValueError(f"Unsupported signature scheme: {signature_scheme}")
    cfg = ProviderConfig(
        master_secret_hex=secret.hex(),
        halo2_prover_dir=halo2_dir.resolve(),
        default_base_image=base_image.resolve() if base_image else None,
        signing_key_hex=signing_key.hex(),
        signing_pubkey_hex=signing_pubkey.hex() if signing_pubkey else None,
        signature_scheme=signature_scheme,
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
        self.robin = MidTrajectoryInjector(config.pattern)
        self.halo2 = Halo2Bindings(config.halo2_prover_dir)
        self.signature_encoder = StableSignatureEncoder(self.litevae, self.extractor)
        self._signer = create_signer(
            config.signature_scheme,
            config.signing_key,
            config.signing_pubkey,
        )

    @property
    def secret(self) -> bytes:
        return self.config.master_secret

    @property
    def signing_key(self) -> bytes | None:
        return self.config.signing_key

    @property
    def signer(self) -> Signer:
        return self._signer
