from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch optional
    torch = None
    nn = None

if torch is not None:

    class ExtractorNet(nn.Module):  # type: ignore[misc]
        def __init__(self, payload_bits: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, payload_bits),
            )

        def forward(self, x):
            feats = self.features(x)
            return self.head(feats)

else:  # pragma: no cover - keeps module importable without torch

    class ExtractorNet:  # type: ignore[misc]
        def __init__(self, payload_bits: int):
            raise RuntimeError("PyTorch is required for ExtractorNet.")


@dataclass
class NeuralExtractor:
    model: ExtractorNet
    device: "torch.device"

    @classmethod
    def is_available(cls) -> bool:
        return torch is not None

    @classmethod
    def from_checkpoint(cls, weights_path: Path, payload_bits: int, device: Optional[str] = None) -> "NeuralExtractor":
        if torch is None:
            raise RuntimeError("PyTorch not installed; install extras with `pip install -e .[train]`.")
        device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = ExtractorNet(payload_bits)
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state["model"])
        model.to(device_obj)
        model.eval()
        return cls(model=model, device=device_obj)

    def predict_bits(self, image: np.ndarray) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch not installed.")
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)
        bits = (probs > 0.5).cpu().numpy().astype(np.uint8).flatten()
        return bits

    def save(self, path: Path) -> None:
        if torch is None:
            raise RuntimeError("PyTorch not installed.")
        payload = {"model": self.model.state_dict()}
        torch.save(payload, path)
