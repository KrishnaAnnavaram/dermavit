"""The single source of truth for the class order.

Training, the confusion matrix, the reports and the ISIC 2018 ground-truth loader all read this
module, so a plot label can never point to another class than the model output.
"""

from __future__ import annotations

CLASSES: tuple[str, ...] = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
DISPLAY = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}
MALIGNANT = ("akiec", "bcc", "mel")
MELANOMA = "mel"
INDEX = {name: i for i, name in enumerate(CLASSES)}
# column names of ISIC2018_Task3_Test_GroundTruth.csv
ISIC_COLUMNS = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AKIEC": "akiec", "BKL": "bkl", "DF": "df", "VASC": "vasc"}


def encode(names) -> list[int]:
    """Map class names to indices. Raises KeyError for an unknown name."""
    out = []
    for n in names:
        if n not in INDEX:
            raise KeyError(f"unknown class {n!r}. Known: {CLASSES}")
        out.append(INDEX[n])
    return out


def decode(indices) -> list[str]:
    return [CLASSES[int(i)] for i in indices]


def melanoma_index() -> int:
    return INDEX[MELANOMA]
