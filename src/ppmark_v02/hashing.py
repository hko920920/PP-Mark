from __future__ import annotations

import cv2
import numpy as np
from scipy.fftpack import dct


def compute_phash(image: np.ndarray, hash_size: int = 16) -> bytes:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    resized = cv2.resize(gray, (hash_size * 2, hash_size * 2), interpolation=cv2.INTER_AREA)
    dct_vals = dct(dct(resized.astype(np.float32), axis=0, norm="ortho"), axis=1, norm="ortho")
    dct_low = dct_vals[:hash_size, :hash_size]
    med = np.median(dct_low[1:, 1:])
    bits = (dct_low.flatten() > med).astype(np.uint8)
    padded = np.pad(bits, (0, (-len(bits)) % 8))
    packed = np.packbits(padded)
    return bytes(packed)
