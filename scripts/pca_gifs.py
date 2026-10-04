"""Make one animated GIF per statement type (and one for all types pooled) that
steps through the layers. Each frame shows a 2-D PCA of the activations at that
layer, with true statements in blue and false statements in orange. If
probe_accuracy.csv is in --out, the accuracy of both classifiers is shown underneath,
with the current layer marked.

The same random sample of statements (fixed seed) is used in every frame. PCA is fit
separately at each layer; its axes are flipped and rotated to line up with the
previous frame, so the cloud moves smoothly instead of jumping around.

Outputs in --out: pca_<type>.gif for every type, and pca_all.gif

Example:
  uv run python scripts/pca_gifs.py --acts activations/smollm2-test --out results/smollm2-test
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import (
    CLASSIFIER_COLORS,
    CLASSIFIER_NAMES,
    LABEL_COLORS,
    MUTED,
    SEED,
    SUBSETS,
    Activations,
    out_dir,
    set_style,
)


def pca_frames(acts: Activations, rows: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """2-D PCA coordinates of `rows` at every layer, aligned frame to frame."""
    frames, prev = [], None
    for layer in range(acts.n_layers):
        X = StandardScaler().fit_transform(acts.layer(layer, rows))
        with np.errstate(invalid="ignore", divide="ignore"):  # identical rows (e.g. layer 0) have zero variance
            Y = PCA(n_components=2, random_state=SEED).fit_transform(X)
        Y = Y / (Y.std() or 1)
        if prev is None:
            # first frame: put true statements on the right
            if Y[y == 1, 0].mean() < Y[y == 0, 0].mean():
                Y[:, 0] = -Y[:, 0]
        else:
            # rotate/flip to best match the previous frame (PCA axes have no fixed sign)
            R, _ = orthogonal_procrustes(Y, prev)
            Y = Y @ R
        frames.append(Y)
        prev = Y
    return frames


def make_gif(acts, subset, rows, probe, path, fps):
    y = acts.statements["label"].to_numpy()[rows]
    frames = pca_frames(acts, rows, y)
    order = np.random.default_rng(SEED).permutation(len(rows))  # mix the colours when drawing
    colors = np.array([LABEL_COLORS[v] for v in y])[order]
    acc = None if probe is None else probe[probe["type"] == subset]

    if acc is not None and len(acc):
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 7.5), gridspec_kw={"height_ratios": [3, 1.2]})
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax2 = None
    name = "all types pooled" if subset == "all" else f"{subset} statements"
    fig.suptitle(f"PCA of {name}, {acts.model_name}", x=0.02, ha="left", fontsize=12)
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", color=LABEL_COLORS[1], label="true"),
                        Line2D([], [], ls="none", marker="o", color=LABEL_COLORS[0], label="false")],
               loc="upper right", ncol=2)

    def draw(layer):
        Y = frames[layer][order]
        lim = np.percentile(np.abs(Y), 99) * 1.15 or 1.0  # all points identical (e.g. layer 0): avoid a 0 range
        ax.clear()
        ax.scatter(Y[:, 0], Y[:, 1], s=10, c=colors, alpha=0.6, lw=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(f"layer {layer}", loc="left")
        if ax2 is not None:
            ax2.clear()
            ax2.axhline(0.5, color=MUTED, lw=1, ls="--")
            for clf in CLASSIFIER_NAMES:
                d = acc[acc["classifier"] == clf].sort_values("layer")
                ax2.plot(d["layer"], d["acc_mean"], color=CLASSIFIER_COLORS[clf], lw=1.6, label=CLASSIFIER_NAMES[clf])
                cur = d[d["layer"] == layer]
                ax2.plot(cur["layer"], cur["acc_mean"], "o", color=CLASSIFIER_COLORS[clf], ms=6)
            ax2.axvline(layer, color=MUTED, lw=1)
            ax2.set_xlim(0, acts.n_layers - 1)
            ax2.set_ylim(0.4, 1.0)
            ax2.set_xlabel("layer")
            ax2.set_ylabel("test accuracy")
            ax2.legend(loc="upper left", fontsize=8)

    anim = FuncAnimation(fig, draw, frames=acts.n_layers)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts", required=True, help="activation folder from extract_activations.py")
    p.add_argument("--out", required=True, help="output folder for the GIFs")
    p.add_argument("--n-sample", type=int, default=1500,
                   help="statements per GIF, sampled once with a fixed seed (default: %(default)s)")
    p.add_argument("--fps", type=float, default=3, help="frames (layers) per second (default: %(default)s)")
    args = p.parse_args()

    set_style()
    acts = Activations(args.acts)
    out = out_dir(args.out)
    try:
        probe = pd.read_csv(out / "probe_accuracy.csv")
    except FileNotFoundError:
        probe = None
        print("no probe_accuracy.csv in --out: the GIFs will show the PCA only (run probe_layers.py first)")

    rng = np.random.default_rng(SEED)
    for subset in SUBSETS:
        rows = acts.subset_rows(subset)
        if len(rows) == 0:
            continue
        if len(rows) > args.n_sample:
            rows = np.sort(rng.choice(rows, size=args.n_sample, replace=False))
        path = out / f"pca_{subset}.gif"
        print(f"{subset}: {len(rows)} statements, {acts.n_layers} layers ...", flush=True)
        make_gif(acts, subset, rows, probe, path, args.fps)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
