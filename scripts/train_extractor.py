#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from ppmark_v02.config import ProviderConfig
from ppmark_v02.training.configs import DataConfig, ExtractorTrainingConfig, TrainingSchedule
from ppmark_v02.training.extractor_trainer import ExtractorTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PP-Mark extractor network W.")
    parser.add_argument("--config", type=Path, required=True, help="Provider config JSON (for secrets + payload size).")
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        required=True,
        help="Root directory containing training images (can be passed multiple times).",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/extractor/latest.pt"), help="Where to save weights.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider_cfg = ProviderConfig.load(args.config)
    data_cfg = DataConfig(dataset_roots=[root.resolve() for root in args.data_root], batch_size=args.batch_size)
    schedule = TrainingSchedule(
        epochs=args.epochs,
        learning_rate=args.lr,
        log_interval=args.log_interval,
    )
    train_cfg = ExtractorTrainingConfig(
        data=data_cfg,
        schedule=schedule,
        payload_bits=provider_cfg.extractor.payload_bits,
    )
    trainer = ExtractorTrainer(train_cfg, provider_cfg)
    trainer.train(args.output)
    print(f"[extractor] Saved weights to {args.output}")


if __name__ == "__main__":
    main()
