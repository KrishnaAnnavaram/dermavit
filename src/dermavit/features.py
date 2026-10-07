"""Handcrafted colour and texture features for the classical baseline (no deep learning).

The baseline gives the transformer models a floor to beat, and it lets the full pipeline (split,
imbalance handling, metrics, threshold) run in CI without torch.
"""

from __future__ import annotations

import numpy as np


def _rgb_to_hsv(x: np.ndarray) -> np.ndarray:
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    diff = mx - mn + 1e-8
    h = np.where(mx == r, ((g - b) / diff) % 6, np.where(mx == g, (b - r) / diff + 2, (r - g) / diff + 4)) / 6.0
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-8), 0.0)
    return np.stack([h, s, mx], -1)


def _moments(values: np.ndarray) -> list[float]:
    m, sd = float(values.mean()), float(values.std())
    skew = float(((values - m) ** 3).mean() / (sd ** 3 + 1e-8))
    return [m, sd, skew]


def image_features(img: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> 1-D float vector (colour moments, hue histogram, centre contrast, texture)."""
    x = img.astype(np.float32) / 255.0
    hsv = _rgb_to_hsv(x)
    h, w = x.shape[:2]
    feats: list[float] = []
    for c in range(3):
        feats += _moments(x[..., c].ravel())
    for c in range(3):
        feats += _moments(hsv[..., c].ravel())
    hist, _ = np.histogram(hsv[..., 0], bins=8, range=(0, 1))
    feats += list(hist / hist.sum())
    cy, cx = slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4)
    centre = x[cy, cx].reshape(-1, 3).mean(0)
    border = np.concatenate([x[: h // 8].reshape(-1, 3), x[-h // 8:].reshape(-1, 3)]).mean(0)
    feats += list(centre - border)
    gray = x.mean(-1)
    gy, gx = np.gradient(gray)
    grad = np.hypot(gx, gy)
    feats += [float(grad.mean()), float(grad.std()), float((gray < 0.35).mean())]
    centre_gray = gray[cy, cx]
    feats += [float(centre_gray.std()), float(np.abs(np.diff(centre_gray, axis=1)).mean())]
    return np.asarray(feats, dtype=np.float32)


def feature_matrix(image_ids, source) -> np.ndarray:
    return np.stack([image_features(source.get(i)) for i in image_ids])


FEATURE_NAMES: list[str] = (
    [f"rgb_{c}_{m}" for c in "rgb" for m in ("mean", "std", "skew")]
    + [f"hsv_{c}_{m}" for c in "hsv" for m in ("mean", "std", "skew")]
    + [f"hue_bin_{i}" for i in range(8)]
    + [f"centre_minus_border_{c}" for c in "rgb"]
    + ["grad_mean", "grad_std", "dark_share", "centre_gray_std", "centre_roughness"]
)
