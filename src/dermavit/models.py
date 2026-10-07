"""Model factory (needs the ``torch`` extra; ``vit_b16`` and ``hiera_base`` need the ``hf`` extra).

Import this module only when you train. The core package never imports it.
Each model returns raw logits of shape (batch, 7) in the class order of ``labels.CLASSES``.
"""

from __future__ import annotations

import torch
from torch import nn

from dermavit.images import spec_for
from dermavit.labels import CLASSES


class TinyCNN(nn.Module):
    """A small CNN for CPU smoke tests and the offline torch demo."""

    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.features = nn.Sequential(block(3, 16), nn.MaxPool2d(2), block(16, 32), nn.MaxPool2d(2), block(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, num_classes))
        self.cam_layer = self.features[-1]

    def forward(self, x):
        return self.head(self.features(x))


class HFLogits(nn.Module):
    """Wrap a Hugging Face image classifier so that ``forward(x)`` returns the logits tensor."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x):
        return self.hf_model(pixel_values=x).logits


def build_model(name: str, num_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    spec = spec_for(name)
    if name == "tiny_cnn":
        return TinyCNN(num_classes)
    if name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0

        model = efficientnet_b0(weights=spec.weights if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model.cam_layer = model.features[-1]
        return model
    try:
        from transformers import AutoConfig, AutoModelForImageClassification
    except ImportError as exc:
        raise ImportError("transformers is not installed. Run: pip install 'dermavit[hf]'") from exc
    id2label = dict(enumerate(CLASSES))
    label2id = {v: k for k, v in id2label.items()}
    if pretrained:
        hf = AutoModelForImageClassification.from_pretrained(
            spec.weights, num_labels=num_classes, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    else:
        cfg = AutoConfig.from_pretrained(spec.weights, num_labels=num_classes, id2label=id2label, label2id=label2id)
        hf = AutoModelForImageClassification.from_config(cfg)
    return HFLogits(hf)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def pick_device(setting: str = "auto") -> torch.device:
    if setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(setting)
