"""
Image processing utilities for MVP
Compute vegetation indices (NDVI, EVI) and basic anomaly maps
"""

import numpy as np
from typing import Dict


def compute_ndvi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute NDVI = (NIR - RED) / (NIR + RED)"""
    nir = bands['nir'].astype(np.float32)
    red = bands['red'].astype(np.float32)
    denom = (nir + red)
    denom[denom == 0] = 1e-6
    ndvi = (nir - red) / denom
    return np.clip(ndvi, -1, 1)


def compute_evi(bands: Dict[str, np.ndarray], G=2.5, C1=6.0, C2=7.5, L=1.0) -> np.ndarray:
    """Compute EVI = G * (NIR - RED) / (NIR + C1*RED - C2*BLUE + L)"""
    nir = bands['nir'].astype(np.float32)
    red = bands['red'].astype(np.float32)
    blue = bands['blue'].astype(np.float32)
    denom = (nir + C1 * red - C2 * blue + L)
    denom[denom == 0] = 1e-6
    evi = G * (nir - red) / denom
    return np.clip(evi, -1, 1)


def zscore_anomaly(img: np.ndarray, thresh: float = 2.0) -> np.ndarray:
    """Return binary anomaly map using z-score thresholding"""
    mean = np.mean(img)
    std = np.std(img) + 1e-6
    z = (img - mean) / std
    return (np.abs(z) > thresh).astype(np.uint8)


def simple_health_score(ndvi: np.ndarray, evi: np.ndarray) -> np.ndarray:
    """Combine indices into a simple 0-1 health score"""
    ndvi_n = (ndvi + 1) / 2  # scale [-1,1] to [0,1]
    evi_n = (evi + 1) / 2
    score = 0.6 * ndvi_n + 0.4 * evi_n
    return np.clip(score, 0, 1)

