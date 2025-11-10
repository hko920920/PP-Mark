import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import cv2
import numpy as np

from dct_watermark import DCTWatermarker, WatermarkConfig


@dataclass
class AttackResult:
    name: str
    success: bool
    error: str | None = None
    notes: str | None = None

    def as_dict(self) -> Dict[str, str]:
        return {
            "attack": self.name,
            "success": str(self.success),
            "error": self.error or "",
            "notes": self.notes or "",
        }


def attack_identity(img: np.ndarray) -> np.ndarray:
    return img.copy()


def attack_jpeg(img: np.ndarray, quality: int = 50) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def attack_gaussian_noise(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def attack_crop(img: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = int(h * crop_ratio)
    new_w = int(w * crop_ratio)
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    cropped = img[y0 : y0 + new_h, x0 : x0 + new_w]
    canvas = np.zeros_like(img)
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = cropped
    return canvas


def run_attack(
    name: str,
    attack_fn: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    tmp_dir: Path,
    watermarker: DCTWatermarker,
    secret: bytes,
) -> AttackResult:
    attacked = attack_fn(image)
    attack_path = tmp_dir / f"{name}.png"
    cv2.imwrite(str(attack_path), attacked)
    try:
        recovered = watermarker.extract(attack_path)
        return AttackResult(name=name, success=recovered == secret)
    except Exception as exc:  # pylint: disable=broad-except
        return AttackResult(name=name, success=False, error=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DCT watermark robustness against common attacks.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("assets/processed/coco/coco_val_street_market.png"),
        help="Path to the clean source image (default: prepared COCO sample).",
    )
    parser.add_argument("--workdir", type=Path, default=Path("pp_mark/results"), help="Directory to store intermediate artifacts.")
    parser.add_argument("--message", type=str, default="PP-Mark Rocks", help="Secret message to embed.")
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=0.8,
        help="Center crop ratio (0-1, lower removes more border).",
    )
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    watermarked_path = args.workdir / "watermarked.png"

    config = WatermarkConfig()
    watermarker = DCTWatermarker(config)

    watermarker.embed(args.input, watermarked_path, args.message.encode("utf-8"))
    watermarked_img = cv2.imread(str(watermarked_path), cv2.IMREAD_COLOR)
    if watermarked_img is None:
        raise FileNotFoundError("Watermarked image generation failed")

    attacks: List[AttackResult] = []
    attacks.append(
        run_attack("clean", attack_identity, watermarked_img, args.workdir, watermarker, args.message.encode("utf-8"))
    )
    attacks.append(
        run_attack("jpeg_q50", lambda img: attack_jpeg(img, 50), watermarked_img, args.workdir, watermarker, args.message.encode("utf-8"))
    )
    attacks.append(
        run_attack(
            "gaussian_noise",
            lambda img: attack_gaussian_noise(img, sigma=12.0),
            watermarked_img,
            args.workdir,
            watermarker,
            args.message.encode("utf-8"),
        )
    )
    attacks.append(
        run_attack(
            "center_crop",
            lambda img: attack_crop(img, crop_ratio=args.crop_ratio),
            watermarked_img,
            args.workdir,
            watermarker,
            args.message.encode("utf-8"),
        )
    )

    attacks.append(
        AttackResult(
            name="mueller_imprint_forgery",
            success=False,
            notes="Not executed here; requires semantic-forgery repo and GPU. Expected CPR ~ 0 per theoretical analysis.",
        )
    )

    table_path = args.workdir / "attack_matrix.json"
    table_path.write_text(json.dumps([a.as_dict() for a in attacks], indent=2))

    print("Attack evaluation summary:")
    for result in attacks:
        status = "PASS" if result.success else "FAIL"
        note = f" ({result.notes})" if result.notes else ""
        err = f" :: {result.error}" if result.error else ""
        print(f"- {result.name}: {status}{note}{err}")


if __name__ == "__main__":
    main()
