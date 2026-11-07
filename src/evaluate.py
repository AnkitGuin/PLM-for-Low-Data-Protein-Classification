"""
evaluate.py
-----------
Computes the four metrics we track for every experiment (accuracy, macro F1,
macro precision, macro recall) and plots confusion matrices for the
full-data (100%) models.

Macro-averaging treats every class equally regardless of how many examples
it has, which matters here since remote homology has many fold classes with
very few examples each.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred):
    """Return metrics, macro-averaging over the labels present in y_true."""
    labels = sorted(set(y_true))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    """
    Plot (and optionally save) a confusion matrix.

    Remote homology has a large number of classes, so we don't label every
    row/column -- the point of this plot is to visually show how concentrated
    (or spread out) the errors are, not to read off individual class names.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_learning_curve(results_df, metric="macro_f1", save_path=None):
    """
    Plot `metric` versus training-data fraction, with one line per representation.
    The input must include representation, fraction, seed, and the metric column.

    If a representation was run with multiple seeds at the same fraction,
    they're averaged and shown with an error bar (std dev).
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for representation, group in results_df.groupby("representation"):
        summary = group.groupby("fraction")[metric].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            summary["fraction"] * 100,
            summary["mean"],
            yerr=summary["std"].fillna(0),
            marker="o",
            capsize=3,
            label=representation,
        )

    ax.set_xscale("log")
    ax.set_xlabel("% of training data used (log scale)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Learning Curve by Representation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    # Quick manual check with synthetic predictions: python src/evaluate.py
    y_true = [0, 1, 2, 2, 1, 0, 2, 1]
    y_pred = [0, 1, 1, 2, 1, 0, 2, 2]

    metrics = compute_metrics(y_true, y_pred)
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    plot_confusion_matrix(y_true, y_pred, title="sanity check")
    print("confusion matrix plotted OK")

    fake_results = pd.DataFrame([
        {"representation": "aa_composition", "fraction": f, "seed": 0, "macro_f1": 0.1 + f * 0.3}
        for f in [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    ] + [
        {"representation": "esm2", "fraction": f, "seed": 0, "macro_f1": 0.3 + f * 0.4}
        for f in [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    ])
    plot_learning_curve(fake_results)
    print("learning curve plotted OK")
