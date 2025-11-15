from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass
class DataConfig:
    dataset_roots: Sequence[Path]
    batch_size: int = 8
    shuffle: bool = True
    crop_ratio_range: Tuple[float, float] = (0.85, 1.0)
    jpeg_quality_range: Tuple[int, int] = (70, 95)
    color_jitter: float = 0.05
    horizontal_flip: bool = True
    seed: Optional[int] = None
    max_samples: Optional[int] = None

    def resolved_roots(self) -> List[Path]:
        return [Path(root).resolve() for root in self.dataset_roots]


@dataclass
class TrainingSchedule:
    epochs: int
    learning_rate: float
    log_interval: int = 10


@dataclass
class ExtractorTrainingConfig:
    data: DataConfig
    schedule: TrainingSchedule
    payload_bits: int
