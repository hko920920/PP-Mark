from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class LiteVAEConfig:
    wavelet: str = "haar"
    levels: int = 2
    conditioning_strength: float = 0.12
    payload_repeats: int = 4
    carrier_seed: int = 1337


@dataclass
class ExtractorConfig:
    tile_size: int = 32
    crop_ratio: float = 0.9
    confidence_threshold: float = 0.6
    repetitions: int = 3
    payload_bits: int = 256


@dataclass
class PatternConfig:
    t_injection: int = 36
    radial_frequency: int = 12
    amplitude: float = 0.07


@dataclass
class ProviderConfig:
    master_secret_hex: str
    halo2_prover_dir: Path
    litevae: LiteVAEConfig = field(default_factory=LiteVAEConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    pattern: PatternConfig = field(default_factory=PatternConfig)
    default_base_image: Path | None = None
    extractor_weights: Path | None = None
    litevae_weights: Path | None = None
    pattern_weights: Path | None = None

    @property
    def master_secret(self) -> bytes:
        return bytes.fromhex(self.master_secret_hex)

    @classmethod
    def load(cls, path: Path) -> "ProviderConfig":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        base_dir = path.parent

        def _resolve(value: str | None) -> Path | None:
            if value is None:
                return None
            p = Path(value)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            return p

        halo2_dir = _resolve(raw["halo2_prover_dir"])
        if halo2_dir is None:
            raise ValueError("halo2_prover_dir missing in config.")
        extractor_weights = _resolve(raw.get("extractor_weights"))
        litevae_weights = _resolve(raw.get("litevae_weights"))
        pattern_weights = _resolve(raw.get("pattern_weights"))

        return cls(
            master_secret_hex=raw["master_secret_hex"],
            halo2_prover_dir=halo2_dir,
            litevae=LiteVAEConfig(**raw.get("litevae", {})),
            extractor=ExtractorConfig(**raw.get("extractor", {})),
            pattern=PatternConfig(**raw.get("pattern", {})),
            default_base_image=(
                _resolve(raw.get("default_base_image"))
            ),
            extractor_weights=extractor_weights,
            litevae_weights=litevae_weights,
            pattern_weights=pattern_weights,
        )

    def dump(self, path: Path) -> None:
        payload: Dict[str, Any] = {
            "master_secret_hex": self.master_secret_hex,
            "halo2_prover_dir": str(self.halo2_prover_dir),
            "litevae": asdict(self.litevae),
            "extractor": asdict(self.extractor),
            "pattern": asdict(self.pattern),
        }
        if self.default_base_image is not None:
            payload["default_base_image"] = str(self.default_base_image)
        if self.extractor_weights is not None:
            payload["extractor_weights"] = str(self.extractor_weights)
        if self.litevae_weights is not None:
            payload["litevae_weights"] = str(self.litevae_weights)
        if self.pattern_weights is not None:
            payload["pattern_weights"] = str(self.pattern_weights)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
