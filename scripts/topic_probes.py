"""Experiments E1-E3 of the report (Section 3.4): layer-wise truth probes per topic.

Uses only affirmative and negated statements, pooled (Section 3.1). At every layer:

  E1  pooled baseline   one probe on all six topics together, 6-fold CV
  E2  per-topic         one probe per topic, 6-fold CV on that topic alone
  E3  transfer          each fold-wise probe of topic t applied to all statements of
                        every other topic t'; accuracy and AUROC averaged over the folds

Each probe is StandardScaler + L2 logistic regression, fitted on the training folds
only. Folds come from StratifiedGroupKFold (grouped by `group`, stratified by label,
shuffled with a fixed seed). They are computed once per dataset (pooled, and each
topic) from the statement ids, so they are identical at every layer and for every
model extracted from the same statement file.

Outputs in --out:
  e1_pooled.csv      layer, rel_depth, n, chance, acc_mean, acc_std, auroc_mean, auroc_std
  e2_per_topic.csv   same, plus topic
  e3_transfer.csv    layer, rel_depth, train_topic, test_topic, acc_*, auroc_*
                     (diagonal = held-out CV values of E2)
  folds.csv          id, dataset, fold: the fold assignment that was used
  e1_e2_depth.png    E1 and E2 accuracy against relative depth
  e3_offdiag.png     mean off-diagonal transfer accuracy and AUROC against depth
  e3_matrices.png    transfer matrices (accuracy and AUROC) at representative depths

Example:
  uv run python scripts/topic_probes.py --acts activations/smollm2-test --out results/smollm2-test
"""

import argparse
import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import MUTED, SEED, Activations, out_dir, set_style

TYPES = ["affirmative", "negated"]
TOPICS = ["animal_class", "cities", "element_symb", "facts", "inventors", "sp_en_trans"]
POOLED = "pooled"
TOPIC_COLORS = dict(zip(TOPICS, ["#2a78d6", "#eb6834", "#1baf7a", "#a35bd6", "#d6a12a", "#d65b8f"]))


def make_folds(df: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped, stratified folds of the rows of df (positions 0..len(df)-1)."""
    df = df.reset_index(drop=True)
    order = np.argsort(df["id"].to_numpy(), kind="stable")  # independent of row order on disk
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    splits = cv.split(order, df["label"].to_numpy()[order], groups=df["group"].to_numpy()[order])
    return [(order[tr], order[te]) for tr, te in splits]


def fit_fold(X, y, train, test, C, max_iter, others):
    """Fit one probe on X[train]; score it on X[test] and on every (X', y') in others."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter, random_state=SEED))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X[train], y[train])
    n_iter = int(clf[-1].n_iter_.max())

    def score(Xe, ye):
        z = clf.decision_function(Xe)
        return float(((z > 0).astype(int) == ye).mean()), float(roc_auc_score(ye, z))

    return score(X[test], y[test]), {t: score(Xo, yo) for t, (Xo, yo) in others.items()}, n_iter


def summarise(scores: list[tuple[float, float]]) -> dict:
    a = np.array(scores)
    return {"acc_mean": a[:, 0].mean(), "acc_std": a[:, 0].std(),
            "auroc_mean": a[:, 1].mean(), "auroc_std": a[:, 1].std()}


def plot_depth(e1, e2, model, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, metric, name in zip(axes, ["acc", "auroc"], ["held-out accuracy", "held-out AUROC"]):
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.grid(axis="y")
        for t in TOPICS:
            d = e2[e2["topic"] == t]
            ax.plot(d["rel_depth"], d[f"{metric}_mean"], color=TOPIC_COLORS[t], lw=1.5, label=t)
        ax.plot(e1["rel_depth"], e1[f"{metric}_mean"], color="#222", lw=2.5, label="pooled (E1)")
        ax.fill_between(e1["rel_depth"], e1[f"{metric}_mean"] - e1[f"{metric}_std"],
                        e1[f"{metric}_mean"] + e1[f"{metric}_std"], color="#222", alpha=0.12, lw=0)
        ax.set_xlabel("relative depth ℓ / L")
        ax.set_ylabel(name)
        ax.set_ylim(0.4, 1.0)
    axes[1].legend(loc="lower right", ncol=2)
    fig.suptitle(f"Truth decodability per topic (E1, E2): {model}", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, -0.02, "Mean over 6 grouped, stratified folds; shaded band = ±1 std of E1. "
             "Dashed line = chance.", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_offdiag(e3, model, path):
    off = e3[e3["train_topic"] != e3["test_topic"]]
    diag = e3[e3["train_topic"] == e3["test_topic"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.grid(axis="y")
    for d, metric, color, ls, label in [
        (diag, "acc", "#222", ":", "within topic, accuracy (diagonal)"),
        (off, "acc", "#2a78d6", "-", "transfer, accuracy"),
        (off, "auroc", "#1baf7a", "-", "transfer, AUROC"),
    ]:
        g = d.groupby("rel_depth")[f"{metric}_mean"]
        m, lo, hi = g.mean(), g.min(), g.max()
        ax.plot(m.index, m.values, color=color, ls=ls, label=label)
        if d is off:
            ax.fill_between(m.index, lo.values, hi.values, color=color, alpha=0.10, lw=0)
    ax.set_xlabel("relative depth ℓ / L")
    ax.set_ylabel("mean over topic pairs")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="upper left")
    ax.set_title(f"Cross-topic transfer (E3): {model}", loc="left")
    fig.text(0.01, -0.03, "Lines = mean over the 30 off-diagonal pairs (or 6 diagonal entries); "
             "shaded = min to max over pairs.", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_matrices(e3, n_blocks, model, path, depths=(0.25, 0.5, 0.75, 1.0)):
    layers = sorted({max(1, int(round(d * n_blocks))) for d in depths})
    fig, axes = plt.subplots(2, len(layers), figsize=(3.4 * len(layers), 6.6), squeeze=False)
    for j, layer in enumerate(layers):
        d = e3[e3["layer"] == layer]
        for i, metric in enumerate(["acc", "auroc"]):
            M = d.pivot(index="train_topic", columns="test_topic", values=f"{metric}_mean").loc[TOPICS, TOPICS]
            ax = axes[i, j]
            ax.imshow(M.values, vmin=0.5, vmax=1.0, cmap="viridis")
            for (r, c), v in np.ndenumerate(M.values):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v < 0.8 else "black")
            ax.set_xticks(range(len(TOPICS)), TOPICS if i == 1 else [], rotation=60, ha="right")
            ax.set_yticks(range(len(TOPICS)), TOPICS if j == 0 else [])
            ax.set_title(f"layer {layer} ({layer / n_blocks:.2f}), {'accuracy' if metric == 'acc' else 'AUROC'}",
                         fontsize=9)
            ax.grid(False)
    fig.suptitle(f"Transfer matrices A_ℓ (rows: train topic, columns: test topic): {model}",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts", required=True, help="activation folder from extract_activations.py")
    p.add_argument("--out", required=True, help="output folder for CSVs and figures")
    p.add_argument("--folds", type=int, default=6, help="cross-validation folds (default: %(default)s)")
    p.add_argument("--C", type=float, default=1.0, help="inverse L2 strength of LogisticRegression (default: %(default)s)")
    p.add_argument("--max-iter", type=int, default=5000, help="lbfgs iterations (default: %(default)s)")
    p.add_argument("--layers", default=None, help="comma-separated layers to run (default: all)")
    p.add_argument("--n-jobs", type=int, default=-1, help="CPU cores to use (default: all)")
    args = p.parse_args()

    set_style()
    acts = Activations(args.acts)
    out = out_dir(args.out)
    n_blocks = acts.n_layers - 1  # L; layer 0 is the embedding output
    st = acts.statements
    keep = np.flatnonzero(st["type"].isin(TYPES).to_numpy())
    D = st.iloc[keep].reset_index(drop=True)
    y = D["label"].to_numpy()
    topic_rows = {t: np.flatnonzero(D["topic"].to_numpy() == t) for t in TOPICS}
    topic_rows = {t: r for t, r in topic_rows.items() if len(r)}

    folds = {POOLED: make_folds(D, args.folds)}
    for t, r in topic_rows.items():
        folds[t] = [(r[tr], r[te]) for tr, te in make_folds(D.iloc[r], args.folds)]
    fold_table = [{"id": D["id"].iat[i], "dataset": name, "fold": k}
                  for name, fs in folds.items() for k, (_, te) in enumerate(fs) for i in te]
    pd.DataFrame(fold_table).sort_values(["dataset", "id"]).to_csv(out / "folds.csv", index=False)

    def chance(rows):
        m = y[rows].mean()
        return max(m, 1 - m)

    layers = range(acts.n_layers) if args.layers is None else [int(x) for x in args.layers.split(",")]
    print(f"{acts.model_name}: {len(D)} affirmative+negated statements, L = {n_blocks} blocks, "
          f"topics: {', '.join(f'{t} ({len(r)})' for t, r in topic_rows.items())}; C = {args.C}", flush=True)

    e1, e2, e3, max_iter_hit = [], [], [], 0
    t0 = time.time()
    for layer in layers:
        X = acts.layer(layer, keep)
        others = {t: (X[r], y[r]) for t, r in topic_rows.items()}
        jobs = [(POOLED, k, tr, te, {}) for k, (tr, te) in enumerate(folds[POOLED])]
        jobs += [(t, k, tr, te, {u: others[u] for u in topic_rows if u != t})
                 for t in topic_rows for k, (tr, te) in enumerate(folds[t])]
        res = Parallel(n_jobs=args.n_jobs)(
            delayed(fit_fold)(X, y, tr, te, args.C, args.max_iter, oth) for _, _, tr, te, oth in jobs)
        max_iter_hit += sum(r[2] >= args.max_iter for r in res)

        base = {"layer": layer, "rel_depth": layer / n_blocks}
        by_ds = {}
        for (name, _, _, _, _), r in zip(jobs, res):
            by_ds.setdefault(name, []).append(r)
        e1.append({**base, "n": len(D), "chance": chance(np.arange(len(D))),
                   **summarise([r[0] for r in by_ds[POOLED]])})
        for t in topic_rows:
            rs = by_ds[t]
            within = summarise([r[0] for r in rs])
            e2.append({**base, "topic": t, "n": len(topic_rows[t]), "chance": chance(topic_rows[t]), **within})
            for u in topic_rows:
                s = within if u == t else summarise([r[1][u] for r in rs])
                e3.append({**base, "train_topic": t, "test_topic": u, **s})
        off = [r["acc_mean"] for r in e3[-len(topic_rows) ** 2:] if r["train_topic"] != r["test_topic"]]
        print(f"layer {layer:02d} ({layer / n_blocks:.2f})  E1 acc {e1[-1]['acc_mean']:.3f}  "
              f"E2 acc " + " ".join(f"{r['acc_mean']:.2f}" for r in e2[-len(topic_rows):]) +
              f"  E3 off-diag acc {np.mean(off):.3f}   [{time.time() - t0:.0f}s]", flush=True)

    e1, e2, e3 = pd.DataFrame(e1), pd.DataFrame(e2), pd.DataFrame(e3)
    e1.to_csv(out / "e1_pooled.csv", index=False)
    e2.to_csv(out / "e2_per_topic.csv", index=False)
    e3.to_csv(out / "e3_transfer.csv", index=False)
    (out / "topic_probes_config.json").write_text(json.dumps({
        "model": acts.meta["model"], "acts": args.acts, "types": TYPES, "n_statements": len(D),
        "n_blocks": n_blocks, "folds": args.folds, "fold_splitter": "StratifiedGroupKFold(shuffle=True)",
        "seed": SEED, "C": args.C, "max_iter": args.max_iter, "fits_hitting_max_iter": int(max_iter_hit),
    }, indent=2))
    if max_iter_hit:
        print(f"WARNING: {max_iter_hit} fits reached max_iter={args.max_iter} without converging")

    plot_depth(e1, e2, acts.model_name, out / "e1_e2_depth.png")
    plot_offdiag(e3, acts.model_name, out / "e3_offdiag.png")
    if args.layers is None:
        plot_matrices(e3, n_blocks, acts.model_name, out / "e3_matrices.png")
    print(f"wrote E1-E3 CSVs and figures to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
