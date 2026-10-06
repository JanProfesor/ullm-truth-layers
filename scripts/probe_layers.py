"""Train two linear classifiers on every layer and report how well they separate
true from false statements.

For every layer and every subset (each statement type, plus all types pooled) two
classifiers are trained and tested with 5-fold cross-validation:

  logreg  StandardScaler + logistic regression
  svm     StandardScaler + linear support vector machine (C=0.01: stronger
          regularisation makes it converge ~30x faster at the same accuracy)

The folds are grouped by the `group` column (GroupKFold), so related statements,
such as a fact and its negation or all statements about one city, are always on the
same side of a train/test split. Layers are loaded one at a time.

Outputs in --out:
  probe_accuracy.csv   layer, type, classifier, n, acc_mean, acc_std
  probe_accuracy.png   one small panel per subset: accuracy vs layer, both classifiers

Example:
  uv run python scripts/probe_layers.py --acts activations/smollm2-test --out results/smollm2-test
"""

import argparse
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import (
    CLASSIFIER_COLORS,
    CLASSIFIER_NAMES,
    MUTED,
    SEED,
    SUBSETS,
    Activations,
    out_dir,
    set_style,
)


def make_classifier(name: str):
    if name == "logreg":
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    else:
        model = LinearSVC(C=0.01, max_iter=5000, random_state=SEED)
    return make_pipeline(StandardScaler(), model)


def plot_accuracy(res: pd.DataFrame, title: str, path) -> None:
    """One small panel per subset, each with one line per classifier."""
    subsets = [s for s in SUBSETS if s in set(res["type"])]
    ncol = 3
    nrow = int(np.ceil(len(subsets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.2 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax, subset in zip(axes.flat, subsets):
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.grid(axis="y")
        for clf in res["classifier"].unique():
            d = res[(res["type"] == subset) & (res["classifier"] == clf)].sort_values("layer")
            ax.plot(d["layer"], d["acc_mean"], color=CLASSIFIER_COLORS[clf], label=CLASSIFIER_NAMES[clf])
            ax.fill_between(d["layer"], d["acc_mean"] - d["acc_std"], d["acc_mean"] + d["acc_std"],
                            color=CLASSIFIER_COLORS[clf], alpha=0.12, lw=0)
        n = res.loc[res["type"] == subset, "n"].iloc[0]
        ax.set_title("all types pooled" if subset == "all" else subset, loc="left")
        ax.text(0.99, 0.03, f"n = {n}", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
        ax.set_ylim(0.4, 1.0)
    for ax in axes.flat[len(subsets):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("layer")
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.text(0.01, -0.01, "Mean accuracy on held-out statements over 5 grouped folds; shaded band = ±1 std. "
             "Dashed line = 0.5 (chance).", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts", required=True, help="activation folder from extract_activations.py")
    p.add_argument("--out", required=True, help="output folder for the CSV and figure")
    p.add_argument("--folds", type=int, default=3, help="number of cross-validation folds (default: %(default)s)")
    p.add_argument("--n-jobs", type=int, default=-1, help="CPU cores to use (default: all)")
    args = p.parse_args()

    set_style()
    acts = Activations(args.acts)
    out = out_dir(args.out)
    df = acts.statements
    subsets = {name: acts.subset_rows(name) for name in SUBSETS if len(acts.subset_rows(name))}
    # the folds depend only on the groups and the seed, so they are identical at every layer
    folds = {}
    for name, rows in subsets.items():
        groups = df["group"].to_numpy()[rows]
        k = min(args.folds, len(np.unique(groups)))
        folds[name] = list(GroupKFold(n_splits=k, shuffle=True, random_state=SEED).split(rows, groups=groups))

    print(f"{acts.model_name}: {len(df)} statements, {acts.n_layers} layers, "
          f"subsets: {', '.join(f'{k} ({len(v)})' for k, v in subsets.items())}")
    records = []
    t0 = time.time()
    for layer in range(acts.n_layers):
        X_layer = acts.layer(layer)
        line = []
        for name, rows in subsets.items():
            X, y = X_layer[rows], df["label"].to_numpy()[rows]
            for clf in ["logreg"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    scores = cross_val_score(make_classifier(clf), X, y, cv=folds[name], n_jobs=args.n_jobs)
                records.append({"layer": layer, "type": name, "classifier": clf, "n": len(rows),
                                "acc_mean": scores.mean(), "acc_std": scores.std()})
            line.append(f"{name[:4]} {records[-1]['acc_mean']:.2f}")
        print(f"layer {layer:02d}  (logreg)      " + "  ".join(line) + f"   [{time.time() - t0:.0f}s]", flush=True)

    res = pd.DataFrame(records)
    res.to_csv(out / "probe_accuracy.csv", index=False)
    plot_accuracy(res, f"How well can a linear classifier tell true from false? {acts.model_name}",
                  out / "probe_accuracy.png")
    best = res.loc[res.groupby(["type", "classifier"])["acc_mean"].idxmax(), ["type", "classifier", "layer", "acc_mean"]]
    print("best layer per subset and classifier:\n" + best.to_string(index=False))
    print(f"wrote {out / 'probe_accuracy.csv'} and {out / 'probe_accuracy.png'}")


if __name__ == "__main__":
    main()
