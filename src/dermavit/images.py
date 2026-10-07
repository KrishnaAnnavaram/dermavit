"""Image sources, model-specific normalisation, and the train and eval transforms.

The transforms work on numpy arrays, so they have tests without torch. The eval transform is
deterministic: resize and normalise only. Random augmentation is in the train transform only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str  # "hf", "torchvision" or "local"
    weights: str
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


# Each model uses the statistics of its own pretraining, from its image processor or weights card.
MODEL_SPECS: dict[str, ModelSpec] = {
    "vit_b16": ModelSpec("vit_b16", "hf", "google/vit-base-patch16-224-in21k", 224, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "hiera_base": ModelSpec("hiera_base", "hf", "facebook/hiera-base-224-in1k-hf", 224, IMAGENET_MEAN, IMAGENET_STD),
    "efficientnet_b0": ModelSpec("efficientnet_b0", "torchvision", "IMAGENET1K_V1", 224, IMAGENET_MEAN, IMAGENET_STD),
    "tiny_cnn": ModelSpec("tiny_cnn", "local", "none (random init)", 64, IMAGENET_MEAN, IMAGENET_STD),
}


def spec_for(name: str) -> ModelSpec:
    if name not in MODEL_SPECS:
        raise KeyError(f"unknown model {name!r}. Known: {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[name]


def normalisation_from_processor(hf_id: str) -> tuple[tuple, tuple]:
    """Read ``image_mean`` and ``image_std`` from the Hugging Face image processor (needs ``transformers``)."""
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(hf_id)
    return tuple(proc.image_mean), tuple(proc.image_std)


class ImageSource(Protocol):
    def get(self, image_id: str) -> np.ndarray: ...


class FolderImageSource:
    """Find ``<image_id>.jpg`` or ``.png`` in one or more folders."""

    def __init__(self, *folders: str | Path):
        self.folders = [Path(f) for f in folders]
        self._index: dict[str, Path] | None = None

    def _build(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for folder in self.folders:
            if folder.is_dir():
                for p in folder.rglob("*"):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        index.setdefault(p.stem, p)
        return index

    def get(self, image_id: str) -> np.ndarray:
        if self._index is None:
            self._index = self._build()
        path = self._index.get(image_id)
        if path is None:
            raise FileNotFoundError(f"image {image_id} is not in {[str(f) for f in self.folders]}")
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)


class MemoryImageSource:
    def __init__(self, images: dict[str, np.ndarray]):
        self.images = images

    def get(self, image_id: str) -> np.ndarray:
        return self.images[image_id]


def resize(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def normalise(img: np.ndarray, mean, std) -> np.ndarray:
    """uint8 HxWx3 -> float32 3xHxW with the model statistics."""
    x = img.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


@dataclass
class EvalTransform:
    """Deterministic: resize to the model size, then normalise. Use it for validation and test."""

    size: int
    mean: tuple
    std: tuple

    def __call__(self, img: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        return normalise(resize(img, self.size), self.mean, self.std)


@dataclass
class TrainTransform:
    """Random flips, 90-degree rotations, a random crop (80-100 %) and brightness jitter (+/-10 %)."""

    size: int
    mean: tuple
    std: tuple
    crop_min: float = 0.8
    jitter: float = 0.1

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = img.shape[:2]
        scale = rng.uniform(self.crop_min, 1.0)
        ch, cw = max(1, int(h * scale)), max(1, int(w * scale))
        top, left = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
        out = img[top:top + ch, left:left + cw]
        if rng.random() < 0.5:
            out = out[:, ::-1]
        if rng.random() < 0.5:
            out = out[::-1, :]
        out = np.rot90(out, k=int(rng.integers(0, 4)))
        out = resize(np.ascontiguousarray(out), self.size).astype(np.float32)
        out = np.clip(out * rng.uniform(1 - self.jitter, 1 + self.jitter), 0, 255).astype(np.uint8)
        return normalise(out, self.mean, self.std)


def transforms_for(model: str, size: int | None = None) -> tuple[TrainTransform, EvalTransform]:
    spec = spec_for(model)
    s = size or spec.image_size
    return TrainTransform(s, spec.mean, spec.std), EvalTransform(s, spec.mean, spec.std)
