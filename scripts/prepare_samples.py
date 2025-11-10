#!/usr/bin/env python3
import json
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib import request

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RAW_DIR = ASSETS / "raw"
PROC_DIR = ASSETS / "processed"
META_PATH = ASSETS / "metadata.json"

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE


@dataclass
class Sample:
    dataset: str
    name: str
    url: str
    description: str
    license_note: str

    @property
    def extension(self) -> str:
        return Path(self.url).suffix or ".jpg"

    @property
    def raw_path(self) -> Path:
        return RAW_DIR / self.dataset / f"{self.name}{self.extension}"

    @property
    def processed_path(self) -> Path:
        return PROC_DIR / self.dataset / f"{self.name}.png"


SAMPLES: List[Sample] = [
    Sample(
        dataset="coco",
        name="coco_train_city_bike",
        url="https://images.cocodataset.org/train2017/000000000009.jpg",
        description="COCO train2017 #000000000009 (urban cyclists).",
        license_note="COCO 2017 dataset, CC BY 4.0.",
    ),
    Sample(
        dataset="coco",
        name="coco_val_street_market",
        url="https://images.cocodataset.org/val2017/000000000139.jpg",
        description="COCO val2017 #000000000139 (street market scene).",
        license_note="COCO 2017 dataset, CC BY 4.0.",
    ),
    Sample(
        dataset="coco",
        name="coco_val_beach",
        url="https://images.cocodataset.org/val2017/000000000285.jpg",
        description="COCO val2017 #000000000285 (beach / umbrellas).",
        license_note="COCO 2017 dataset, CC BY 4.0.",
    ),
    Sample(
        dataset="celeba",
        name="celeba_mask_sample",
        url="https://raw.githubusercontent.com/switchablenorms/CelebAMask-HQ/master/images/sample.png",
        description="CelebAMask-HQ sample collage (high-quality face crops).",
        license_note="CelebAMask-HQ sample, research use only.",
    ),
]


def download(sample: Sample) -> None:
    sample.raw_path.parent.mkdir(parents=True, exist_ok=True)
    if sample.raw_path.exists():
        return
    print(f"Downloading {sample.url} -> {sample.raw_path}")
    with request.urlopen(sample.url, context=ssl_ctx) as resp:
        data = resp.read()
    sample.raw_path.write_bytes(data)


def process(sample: Sample) -> None:
    sample.processed_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(sample.raw_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read {sample.raw_path}")
    h, w = img.shape[:2]
    side = min(h, w)
    top = max((h - side) // 2, 0)
    left = max((w - side) // 2, 0)
    cropped = img[top : top + side, left : left + side]
    resized = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(sample.processed_path), resized)


def write_metadata(samples: List[Sample]) -> None:
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample.dataset, []).append(
            {
                "name": sample.name,
                "raw": str(sample.raw_path.relative_to(ASSETS)),
                "processed": str(sample.processed_path.relative_to(ASSETS)),
                "description": sample.description,
                "source_url": sample.url,
                "license": sample.license_note,
            }
        )
    META_PATH.write_text(json.dumps(grouped, indent=2))


def main() -> None:
    for sample in SAMPLES:
        download(sample)
        process(sample)
    write_metadata(SAMPLES)
    print(f"Prepared {len(SAMPLES)} samples. Metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
