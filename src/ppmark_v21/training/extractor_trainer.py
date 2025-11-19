from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ProviderConfig
from ..crypto import poseidon_payload
from ..payload import bytes_to_bits
from ..semantic_anchor import compute_semantic_anchor
from ..models.extractor_net import ExtractorNet
from .configs import ExtractorTrainingConfig
from .datasets import ImageSample, build_image_stream

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - torch optional
    raise RuntimeError("PyTorch is required for training. Install extras via `pip install -e .[train]`.") from exc


@dataclass
class ExtractorTrainer:
    cfg: ExtractorTrainingConfig
    provider_cfg: ProviderConfig

    def __post_init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload_bits = self.provider_cfg.extractor.payload_bits
        self.model = ExtractorNet(payload_bits).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.schedule.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        self.stream = build_image_stream(self.cfg.data)
        self.payload_bits = payload_bits
        self.secret = self.provider_cfg.master_secret
        self.sample_counter = 0

    def train(self, output_path: Path) -> None:
        for epoch in range(self.cfg.schedule.epochs):
            avg_loss = self._train_epoch(epoch)
            print(f"[extractor] epoch={epoch+1} loss={avg_loss:.4f}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": self.model.state_dict(), "payload_bits": self.payload_bits},
            output_path,
        )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        batches = math.ceil(self.stream.num_samples / self.stream.batch_size)
        for step, batch in enumerate(self.stream.iter_epoch()):
            images = torch.stack([self._to_tensor(sample.image) for sample in batch]).to(self.device)
            targets = torch.stack([self._payload_bits(sample) for sample in batch]).to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (step + 1) % self.cfg.schedule.log_interval == 0:
                print(f"[extractor] epoch={epoch+1} step={step+1}/{batches} loss={loss.item():.4f}")
        return total_loss / batches

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor

    def _payload_bits(self, sample: ImageSample) -> torch.Tensor:
        anchor = self._synthetic_anchor(sample)
        payload = poseidon_payload(self.secret, anchor)
        bits = bytes_to_bits(payload)
        if len(bits) < self.payload_bits:
            bits = np.pad(bits, (0, self.payload_bits - len(bits)))
        else:
            bits = bits[: self.payload_bits]
        return torch.from_numpy(bits.astype(np.float32))

    def _synthetic_anchor(self, sample: ImageSample) -> bytes:
        # Use deterministic metadata derived from the sample path/index.
        prompt = sample.path.stem or "ppmark-train"
        seed = f"{sample.index}-{self.sample_counter}"
        model_id = "ppmark-v21-train"
        timestamp = f"2025-01-07T00:{sample.index % 60:02d}:00Z"
        parent_hash = hashlib.sha256(str(sample.path).encode("utf-8")).hexdigest()
        self.sample_counter += 1
        return compute_semantic_anchor(prompt, seed, model_id, timestamp, parent_hash)
