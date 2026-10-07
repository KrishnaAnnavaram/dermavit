"""Synthetic dermatoscopy-like data: metadata, small images and an external test set.

Each lesion has its own colour, size and texture, and 1 to 4 images that differ only by a small
rotation, flip, shift and noise, as in HAM10000. This makes the lesion leakage of an image-level
split measurable offline. The class shares follow HAM10000 (about 67 % nevus).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dermavit.labels import CLASSES, ISIC_COLUMNS

CLASS_SHARE = {"akiec": 0.033, "bcc": 0.051, "bkl": 0.110, "df": 0.012, "mel": 0.111, "nv": 0.669, "vasc": 0.014}
# lesion colour (RGB, 0-255), lesion radius share, texture strength
_PROFILE = {
    "akiec": ((170, 95, 85), 0.30, 18.0),
    "bcc": ((185, 120, 125), 0.26, 10.0),
    "bkl": ((140, 100, 70), 0.33, 22.0),
    "df": ((150, 110, 95), 0.20, 8.0),
    "mel": ((75, 50, 45), 0.36, 30.0),
    "nv": ((120, 80, 60), 0.28, 9.0),
    "vasc": ((170, 50, 70), 0.22, 6.0),
}
LOCATIONS = ("back", "lower extremity", "trunk", "upper extremity", "abdomen", "face", "chest")


def _render(rng: np.random.Generator, colour, radius: float, texture: float, size: int, skin) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    cx, cy = 0.5 + rng.normal(0, 0.04), 0.5 + rng.normal(0, 0.04)
    ax, ay = radius * rng.uniform(0.85, 1.15), radius * rng.uniform(0.85, 1.15)
    dist = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
    mask = np.clip(1.3 - dist, 0, 1)[..., None]
    noise = rng.normal(0, texture, (size, size, 1)) * mask
    img = np.asarray(skin, np.float32) * (1 - mask) + np.asarray(colour, np.float32) * mask + noise
    img += rng.normal(0, 4.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_dataset(n_lesions: int = 600, size: int = 48, seed: int = 0, n_test: int = 150):
    """Return ``(meta, images, test_meta, test_images)``. Images are uint8 arrays keyed by image id."""
    rng = np.random.default_rng(seed)
    names = list(CLASS_SHARE)
    probs = np.asarray([CLASS_SHARE[n] for n in names])
    probs = probs / probs.sum()
    rows, images = [], {}
    image_no = 0

    def lesion_images(dx, n_images):
        nonlocal image_no
        base, radius, texture = _PROFILE[dx]
        colour = np.clip(np.asarray(base) + rng.normal(0, 40, 3), 0, 255)  # lesion-specific colour
        skin = np.clip(np.asarray((220, 180, 160)) + rng.normal(0, 15, 3), 0, 255)
        lesion_seed = int(rng.integers(0, 2**31))
        out = []
        for _ in range(n_images):
            img_rng = np.random.default_rng(lesion_seed)  # same lesion shape for each image
            img = _render(img_rng, colour, radius, texture, size, skin)
            img = np.rot90(img, k=int(rng.integers(0, 4)))
            if rng.random() < 0.5:
                img = img[:, ::-1]
            img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
            image_id = f"ISIC_{image_no:07d}"
            image_no += 1
            images[image_id] = np.ascontiguousarray(img)
            out.append(image_id)
        return out

    for lesion_no in range(n_lesions):
        dx = names[rng.choice(len(names), p=probs)] if lesion_no >= len(names) else names[lesion_no]
        n_images = int(rng.choice([1, 1, 2, 2, 3, 4]))
        sex = rng.choice(["male", "female", "unknown"], p=[0.53, 0.45, 0.02])
        age = float(np.clip(np.round(rng.normal(52, 16) / 5) * 5, 0, 85))
        loc = rng.choice(LOCATIONS)
        dx_type = "histo" if dx in ("mel", "bcc", "akiec") else rng.choice(["histo", "follow_up", "consensus"])
        for image_id in lesion_images(dx, n_images):
            rows.append({"lesion_id": f"HAM_{lesion_no:07d}", "image_id": image_id, "dx": dx, "dx_type": dx_type,
                         "age": age, "sex": sex, "localization": loc})
    meta = pd.DataFrame(rows)

    test_rows, test_images = [], {}
    for _ in range(n_test):
        dx = names[rng.choice(len(names), p=probs)]
        ids = lesion_images(dx, 1)
        test_images[ids[0]] = images.pop(ids[0])
        test_rows.append({"image_id": ids[0], "dx": dx})
    return meta, images, pd.DataFrame(test_rows), test_images


def write_dataset(out_dir: str | Path, n_lesions: int = 600, size: int = 48, seed: int = 0, n_test: int = 150) -> Path:
    """Write the synthetic set in the HAM10000 / ISIC 2018 layout that the CLI reads."""
    out = Path(out_dir)
    meta, images, test_meta, test_images = make_dataset(n_lesions, size, seed, n_test)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "ISIC2018_Task3_Test_Images").mkdir(parents=True, exist_ok=True)
    meta.to_csv(out / "HAM10000_metadata.csv", index=False)
    for image_id, img in images.items():
        Image.fromarray(img).save(out / "images" / f"{image_id}.png")
    for image_id, img in test_images.items():
        Image.fromarray(img).save(out / "ISIC2018_Task3_Test_Images" / f"{image_id}.png")
    gt = pd.DataFrame({"image": test_meta["image_id"]})
    for col, name in ISIC_COLUMNS.items():
        gt[col] = (test_meta["dx"] == name).astype(float)
    gt.to_csv(out / "ISIC2018_Task3_Test_GroundTruth.csv", index=False)
    return out


assert set(CLASS_SHARE) == set(CLASSES)
