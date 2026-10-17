"""Compare the E1-E3 results of several models (output folders of topic_probes.py)
against relative depth, for Section 4 of the report.

Outputs in --out:
  compare_e1.png        pooled accuracy (E1) per model
  compare_e2.png        per-topic accuracy (E2), one panel per model
  compare_e3.png        mean off-diagonal transfer accuracy and AUROC (E3) per model
  compare_summary.csv   per model: emergence depths, peaks and transfer at the peak
  compare_e3_centering.png  only with --centered: E3 transfer with and without topic
                        centering, one panel per model

Example:
  uv run python scripts/compare_models.py --results results/llama3-8b-instruct \
      results/mistral-7b-instruct-v0.3 results/qwen2.5-7b-instruct --out results/comparison       --centered results/llama3-8b-instruct-centered results/mistral-7b-instruct-v0.3-centered       results/qwen2.5-7b-instruct-centered
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import MUTED, out_dir, set_style
from topic_probes import TOPIC_COLORS, TOPICS

MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#a35bd6"]


def load(folder: Path) -> dict:
    cfg = json.loads((folder / "topic_probes_config.json").read_text())
    name = cfg["model"].rstrip("/\\").replace("\\", "/").split("/")[-1]
    if cfg.get("center_topics"):
        name += " (centered)"
    # layer 0 is the embedding output; the report uses layers 1..L
    read = lambda f: pd.read_csv(folder / f).query("layer > 0")  # noqa: E731
    return {"name": name, "e1": read("e1_pooled.csv"), "e2": read("e2_per_topic.csv"),
            "e3": read("e3_transfer.csv")}


def first_depth(d: pd.DataFrame, threshold: float) -> float:
    """Smallest relative depth at which acc_mean reaches the threshold (NaN if never)."""
    hit = d[d["acc_mean"] >= threshold]
    return float(hit["rel_depth"].min()) if len(hit) else float("nan")


def offdiag(e3: pd.DataFrame) -> pd.DataFrame:
    """Mean of the off-diagonal entries of A_l per relative depth."""
    return e3[e3["train_topic"] != e3["test_topic"]].groupby("rel_depth")[["acc_mean", "auroc_mean"]].mean()


def plot_centering(models, centered, colors, path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True, squeeze=False)
    for ax, m, c in zip(axes[0], models, centered):
        col = colors[m["name"]]
        off, offc = offdiag(m["e3"]), offdiag(c["e3"])
        diag = m["e3"][m["e3"]["train_topic"] == m["e3"]["test_topic"]].groupby("rel_depth")["acc_mean"].mean()
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.grid(axis="y")
        ax.plot(diag.index, diag.values, color="#222", ls=":", lw=1.3, label="within topic, accuracy")
        ax.plot(off.index, off["auroc_mean"], color=col, ls="--", lw=1.3, label="transfer, AUROC")
        ax.plot(offc.index, offc["acc_mean"], color=col, label="transfer, accuracy, topic-centered")
        ax.plot(off.index, off["acc_mean"], color=col, alpha=0.45, label="transfer, accuracy")
        ax.fill_between(off.index, off["acc_mean"], offc["acc_mean"].reindex(off.index), color=col, alpha=0.12, lw=0)
        ax.set(xlabel="relative depth ℓ / L", ylim=(0.4, 1.0), xlim=(0, 1))
        ax.set_title(m["name"], loc="left")
    axes[0, 0].set_ylabel("mean over topic pairs")
    axes[0, -1].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Cross-topic transfer (E3) with and without topic centering", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, -0.03, "Shaded = accuracy recovered by centering each topic at its mean (the offset between topics). "
             "The gap that remains to the within-topic line is a difference in direction.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", nargs="+", required=True, help="topic_probes.py output folders")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--centered", nargs="+", default=None,
                   help="topic_probes.py --center-topics output folders, in the same order as --results")
    args = p.parse_args()
    if args.centered and len(args.centered) != len(args.results):
        p.error("--centered needs one folder per --results folder")

    set_style()
    models = [load(Path(r)) for r in args.results]
    centered = [load(Path(r)) for r in args.centered] if args.centered else None
    out = out_dir(args.out)
    colors = dict(zip([m["name"] for m in models], MODEL_COLORS))

    # E1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.grid(axis="y")
    for m in models:
        e1 = m["e1"]
        ax.plot(e1["rel_depth"], e1["acc_mean"], color=colors[m["name"]], label=m["name"])
        ax.fill_between(e1["rel_depth"], e1["acc_mean"] - e1["acc_std"], e1["acc_mean"] + e1["acc_std"],
                        color=colors[m["name"]], alpha=0.12, lw=0)
    ax.set(xlabel="relative depth ℓ / L", ylabel="held-out accuracy", ylim=(0.4, 1.0), xlim=(0, 1))
    ax.legend(loc="lower right")
    ax.set_title("Pooled truth probe (E1)", loc="left")
    fig.tight_layout()
    fig.savefig(out / "compare_e1.png")
    plt.close(fig)

    # E2
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.grid(axis="y")
        for t in TOPICS:
            d = m["e2"][m["e2"]["topic"] == t]
            ax.plot(d["rel_depth"], d["acc_mean"], color=TOPIC_COLORS[t], lw=1.5, label=t)
        ax.plot(m["e1"]["rel_depth"], m["e1"]["acc_mean"], color="#222", lw=2.2, ls=":", label="pooled (E1)")
        ax.set(xlabel="relative depth ℓ / L", ylim=(0.3, 1.0), xlim=(0, 1))
        ax.set_title(m["name"], loc="left")
    axes[0, 0].set_ylabel("held-out accuracy")
    axes[0, -1].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Per-topic truth probes (E2)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "compare_e2.png")
    plt.close(fig)

    # E3
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    rows = []
    for m in models:
        e3 = m["e3"]
        off = e3[e3["train_topic"] != e3["test_topic"]].groupby("rel_depth")[["acc_mean", "auroc_mean"]].mean()
        diag = e3[e3["train_topic"] == e3["test_topic"]].groupby("rel_depth")[["acc_mean", "auroc_mean"]].mean()
        for ax, metric in zip(axes, ["acc_mean", "auroc_mean"]):
            ax.plot(off.index, off[metric], color=colors[m["name"]], label=f"{m['name']}, transfer")
            ax.plot(diag.index, diag[metric], color=colors[m["name"]], ls=":", lw=1.3,
                    label=f"{m['name']}, within topic")

        e1, e2 = m["e1"], m["e2"]
        peak = e1.loc[e1["acc_mean"].idxmax()]
        off_peak = off["acc_mean"].idxmax()
        row = {"model": m["name"], "n_blocks": int(round(e1["layer"].max() / e1["rel_depth"].max())),
               "e1_depth_acc>=0.75": first_depth(e1, 0.75), "e1_depth_acc>=0.95": first_depth(e1, 0.95),
               "e1_peak_acc": peak["acc_mean"], "e1_peak_depth": peak["rel_depth"]}
        for t in TOPICS:
            d = e2[e2["topic"] == t]
            row[f"e2_{t}_depth_acc>=0.75"] = first_depth(d, 0.75)
            row[f"e2_{t}_peak_acc"] = d["acc_mean"].max()
        row.update({"e3_offdiag_peak_acc": off.loc[off_peak, "acc_mean"], "e3_offdiag_peak_depth": off_peak,
                    "e3_offdiag_auroc_at_peak": off.loc[off_peak, "auroc_mean"],
                    "e3_offdiag_peak_auroc": off["auroc_mean"].max(),
                    "e3_offdiag_peak_auroc_depth": off["auroc_mean"].idxmax(),
                    "e3_offdiag_acc_last": off["acc_mean"].iloc[-1],
                    "e3_offdiag_auroc_last": off["auroc_mean"].iloc[-1]})
        rows.append(row)
    for ax, name in zip(axes, ["accuracy", "AUROC"]):
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.grid(axis="y")
        ax.set(xlabel="relative depth ℓ / L", ylim=(0.4, 1.0), xlim=(0, 1))
        ax.set_title(f"Cross-topic transfer (E3), {name}", loc="left")
    axes[0].set_ylabel("mean over topic pairs")
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.text(0.01, -0.03, "Solid = mean of the 30 off-diagonal entries of A_ℓ; dotted = mean of the 6 diagonal "
             "(within-topic, cross-validated) entries.", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "compare_e3.png")
    plt.close(fig)

    if centered:
        plot_centering(models, centered, colors, out / "compare_e3_centering.png")
        for row, c in zip(rows, centered):
            off = offdiag(c["e3"])
            row.update({"e3c_offdiag_peak_acc": off["acc_mean"].max(),
                        "e3c_offdiag_peak_depth": off["acc_mean"].idxmax(),
                        "e3c_offdiag_acc_last": off["acc_mean"].iloc[-1]})

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "compare_summary.csv", index=False)
    print(summary.set_index("model").T.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"wrote compare_e1.png, compare_e2.png, compare_e3.png and compare_summary.csv to {out}")


if __name__ == "__main__":
    main()
