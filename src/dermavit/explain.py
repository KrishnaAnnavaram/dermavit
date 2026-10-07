"""Grad-CAM for convolutional models (needs the ``torch`` extra).

For ViT and Hiera, use attention rollout or a Grad-CAM variant with a reshape step. This module
gives the CNN version, which works for ``tiny_cnn`` and ``efficientnet_b0`` (attribute ``cam_layer``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def grad_cam(model: torch.nn.Module, x: torch.Tensor, class_idx: int | None = None, layer=None) -> np.ndarray:
    """Return a heat map in [0, 1] with the spatial size of ``x`` (shape H x W) for one image."""
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("x must have shape (1, 3, H, W)")
    layer = layer if layer is not None else getattr(model, "cam_layer", None)
    if layer is None:
        raise ValueError("the model has no cam_layer. Pass a layer.")
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        model.eval()
        logits = model(x)
        k = int(logits.argmax(1)) if class_idx is None else int(class_idx)
        model.zero_grad()
        logits[0, k].backward()
    finally:
        h1.remove()
        h2.remove()
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    peak = float(cam.max())
    return (cam / peak).detach().cpu().numpy() if peak > 0 else cam.detach().cpu().numpy()
