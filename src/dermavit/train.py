"""Training loop for the deep models (needs the ``torch`` extra).

Rules that the loop enforces:

- seeds for ``random``, numpy, torch and the data order; the train augmentation is seeded per epoch and image
- random augmentation only for training rows; validation and test use the deterministic eval transform
- class-weighted cross-entropy (or a balanced sampler) for the 67 % nevus imbalance
- the best epoch by validation balanced accuracy is saved and RELOADED before the test evaluation
- each epoch records its time (also the epoch that stops the run), its peak GPU memory and its throughput
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dermavit.imbalance import apply_melanoma_threshold, class_weights, sample_weights, tune_melanoma_threshold
from dermavit.images import transforms_for
from dermavit.labels import encode
from dermavit.metrics import classification_report
from dermavit.models import build_model, count_parameters, pick_device


@dataclass
class TrainConfig:
    model: str = "tiny_cnn"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.05
    patience: int = 3
    seed: int = 42
    weighting: str = "loss"  # "loss" (class-weighted CE), "sampler" (balanced sampler) or "none"
    image_size: int | None = None
    amp: bool = True
    pretrained: bool = True
    device: str = "auto"
    target_recall: float = 0.9
    out_dir: str = "runs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageDataset(Dataset):
    def __init__(self, image_ids, labels, source, transform, train: bool, seed: int = 0):
        self.image_ids, self.labels = list(image_ids), np.asarray(labels, dtype=np.int64)
        self.source, self.transform, self.train, self.seed = source, transform, train, seed
        self.epoch = 0

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.source.get(self.image_ids[idx])
        rng = np.random.default_rng([self.seed, self.epoch, idx]) if self.train else None
        x = self.transform(img, rng)
        return torch.from_numpy(x), int(self.labels[idx])


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x.to(device)).float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 7))


def _peak_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    return None  # not measured on CPU or MPS


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_balanced_accuracy: float
    train_seconds: float
    epoch_seconds: float
    train_images_per_s: float
    peak_memory_mb: float | None
    best: bool


@dataclass
class RunResult:
    model: str
    seed: int
    config: dict
    history: list[EpochLog] = field(default_factory=list)
    best_epoch: int = -1
    reloaded_best: bool = False
    threshold: float = 0.5
    val: dict | None = None
    test: dict | None = None
    test_thresholded: dict | None = None
    efficiency: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["history"] = [asdict(h) for h in self.history]
        return d


def train_model(cfg: TrainConfig, split: pd.DataFrame, source, test_meta: pd.DataFrame | None = None,
                test_source=None) -> RunResult:
    """``split`` is the metadata with a ``split`` column (``train`` / ``val``)."""
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    train_tf, eval_tf = transforms_for(cfg.model, cfg.image_size)
    tr, va = split[split["split"] == "train"], split[split["split"] == "val"]
    y_tr, y_va = encode(tr["dx"]), encode(va["dx"])
    ds_tr = ImageDataset(tr["image_id"], y_tr, source, train_tf, train=True, seed=cfg.seed)
    ds_va = ImageDataset(va["image_id"], y_va, source, eval_tf, train=False)
    gen = torch.Generator().manual_seed(cfg.seed)
    if cfg.weighting == "sampler":
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights(y_tr), dtype=torch.double), len(y_tr),
                                        replacement=True, generator=gen)
        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, sampler=sampler)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size)

    model = build_model(cfg.model, pretrained=cfg.pretrained).to(device)
    weight = torch.as_tensor(class_weights(y_tr), dtype=torch.float32, device=device) if cfg.weighting == "loss" else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = Path(cfg.out_dir) / cfg.model / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "best.pt"
    result = RunResult(cfg.model, cfg.seed, asdict(cfg))
    best, bad_epochs = -1.0, 0
    for epoch in range(cfg.epochs):
        ds_tr.epoch = epoch
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach()) * len(y)
            n += len(y)
        sched.step()
        t_train = time.perf_counter() - t0
        p_va = predict_probs(model, dl_va, device)
        val_bal = classification_report(y_va, p_va)["balanced_accuracy"]
        is_best = val_bal > best
        if is_best:
            best, bad_epochs, result.best_epoch = val_bal, 0, epoch
            torch.save(model.state_dict(), ckpt)
        else:
            bad_epochs += 1
        result.history.append(EpochLog(epoch, total / max(1, n), val_bal, t_train, time.perf_counter() - t0,
                                       n / t_train if t_train > 0 else float("nan"), _peak_mb(device), is_best))
        if bad_epochs >= cfg.patience:
            break  # this epoch is already in the history

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    result.reloaded_best = True
    p_va = predict_probs(model, dl_va, device)
    result.val = classification_report(y_va, p_va, groups=va["lesion_id"].to_numpy(), seed=cfg.seed)
    result.threshold = tune_melanoma_threshold(y_va, p_va, cfg.target_recall)
    if test_meta is not None and len(test_meta):
        y_te = encode(test_meta["dx"])
        ds_te = ImageDataset(test_meta["image_id"], y_te, test_source or source, eval_tf, train=False)
        t0 = time.perf_counter()
        p_te = predict_probs(model, DataLoader(ds_te, batch_size=cfg.batch_size), device)
        eval_s = time.perf_counter() - t0
        result.test = classification_report(y_te, p_te, seed=cfg.seed)
        result.test_thresholded = classification_report(y_te, p_te, apply_melanoma_threshold(p_te, result.threshold))
        result.efficiency["eval_images_per_s"] = len(y_te) / eval_s if eval_s > 0 else None
    peaks = [h.peak_memory_mb for h in result.history if h.peak_memory_mb is not None]
    result.efficiency.update({
        "parameters": count_parameters(model),
        "device": device.type,
        "amp": use_amp,
        "batch_size": cfg.batch_size,
        "image_size": train_tf.size,
        "peak_memory_mb": max(peaks) if peaks else None,
        "train_images_per_s": float(np.mean([h.train_images_per_s for h in result.history])),
        "epochs_run": len(result.history),
        "total_train_seconds": float(sum(h.epoch_seconds for h in result.history)),
    })
    (run_dir / "run.json").write_text(json.dumps(result.to_dict(), indent=2, default=float), encoding="utf-8")
    return result
