"""dermavit: a leakage-free benchmark of ViT, Hiera and CNN baselines for skin-lesion classification.

The core package (splits, transforms, metrics, imbalance handling, the classical baseline) needs no
deep-learning framework. ``dermavit.models``, ``dermavit.train`` and ``dermavit.explain`` need torch.
This is a research benchmark, not a medical device.
"""

__version__ = "0.1.0"
