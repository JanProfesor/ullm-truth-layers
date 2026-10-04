"""Shared helpers for the analysis scripts: loading activations, the fixed seed,
colours and plot style.

The scripts import this module with `from common import ...`, which works because
Python puts the script's own folder (scripts/) on the import path.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEED = 0

TYPE_ORDER = ["affirmative", "negated", "conjunction", "disjunction", "comparison"]
SUBSETS = [*TYPE_ORDER, "all"]  # the five types plus all types pooled

LABEL_COLORS = {1: "#2a78d6", 0: "#eb6834"}  # true = blue, false = orange
CLASSIFIER_COLORS = {"logreg": "#2a78d6", "svm": "#1baf7a"}
CLASSIFIER_NAMES = {"logreg": "Logistic regression", "svm": "Linear SVM"}
MUTED = "#898781"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": "#52514e",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#e1e0d9",
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


class Activations:
    """Reads a folder written by extract_activations.py. Layers are loaded one at a
    time with `layer(i)` (as float32), so memory use stays low."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not (self.path / "meta.json").exists():
            raise SystemExit(f"{self.path} has no meta.json; run extract_activations.py first")
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.statements = pd.read_csv(self.path / "statements.csv")
        self.n_layers = self.meta["n_layers"]  # embedding output + transformer blocks
        self.model_name = self.meta["model"].rstrip("/\\").replace("\\", "/").split("/")[-1]

    def layer(self, i: int, rows=None) -> np.ndarray:
        x = np.load(self.path / f"layer_{i:02d}.npy", mmap_mode="r")
        if rows is not None:
            x = x[rows]
        return np.asarray(x, dtype=np.float32)

    def subset_rows(self, name: str) -> np.ndarray:
        """Row indices of one statement type, or of all statements for 'all'."""
        if name == "all":
            return np.arange(len(self.statements))
        return np.flatnonzero(self.statements["type"].to_numpy() == name)


def out_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
