"""Run the amino-acid-composition low-data baseline on official TAPE splits."""

import argparse
from pathlib import Path
import sys

import pandas as pd
from threadpoolctl import threadpool_limits

try:  # Supports both `python -m src.run_baseline` and direct script execution.
    from .data import TEST_SPLIT, load_train_val_test
    from .evaluate import compute_metrics, plot_confusion_matrix, plot_learning_curve
    from .features import compute_composition_features
    from .train import sample_fraction, train_logistic_regression
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import TEST_SPLIT, load_train_val_test
    from evaluate import compute_metrics, plot_confusion_matrix, plot_learning_curve
    from features import compute_composition_features
    from train import sample_fraction, train_logistic_regression


DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
DEFAULT_SEEDS = (0, 1, 2)


def run_experiment(fractions=DEFAULT_FRACTIONS, seeds=DEFAULT_SEEDS):
    (train_x, train_y), _, (test_x, test_y) = load_train_val_test()
    X_train = compute_composition_features(train_x)
    X_test = compute_composition_features(test_x)

    results = []
    full_model_predictions = None
    # Restrict BLAS to one thread to avoid large thread pools and make this
    # medium-sized multinomial fit more predictable on shared machines.
    with threadpool_limits(limits=1):
        for seed in seeds:
            for fraction in fractions:
                X_sub, y_sub = sample_fraction(X_train, train_y, fraction, seed)
                model = train_logistic_regression(X_sub, y_sub, seed)
                predictions = model.predict(X_test)
                row = compute_metrics(test_y, predictions)
                row.update({
                    "representation": "aa_composition",
                    "fraction": fraction,
                    "seed": seed,
                    "n_train_examples": len(y_sub),
                    "n_train_classes": len(set(y_sub)),
                    "n_test_classes": len(set(test_y)),
                    "n_test_classes_unseen_in_train": len(set(test_y) - set(y_sub)),
                    "classifier_n_iter": int(model.n_iter_.max()),
                    "converged": bool(model.n_iter_.max() < model.max_iter),
                })
                results.append(row)
                print(
                    f"seed={seed} fraction={fraction:>5.0%} n={len(y_sub):>5} "
                    f"classes={row['n_train_classes']:>4} accuracy={row['accuracy']:.4f} "
                    f"macro_f1={row['macro_f1']:.4f} converged={row['converged']}",
                    flush=True,
                )
                if fraction == 1.0 and seed == seeds[0]:
                    full_model_predictions = predictions

    return pd.DataFrame(results), test_y, full_model_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--output", type=Path, default=Path("results/metrics.csv"))
    args = parser.parse_args()

    if not args.seeds:
        parser.error("at least one seed is required")
    if any(not 0 < f <= 1 for f in args.fractions):
        parser.error("fractions must be greater than 0 and at most 1")

    results, y_test, full_predictions = run_experiment(args.fractions, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.exists():
        existing = pd.read_csv(args.output)
        existing = existing[existing["representation"] != "aa_composition"]
        results = pd.concat([existing, results], ignore_index=True)
    results.to_csv(args.output, index=False)

    curve_path = args.output.parent / "learning_curve.png"
    curve_fig = plot_learning_curve(results, save_path=curve_path)
    import matplotlib.pyplot as plt
    plt.close(curve_fig)
    print(f"Saved learning curve to {curve_path}")

    if full_predictions is not None:
        cm_path = args.output.parent / "confusion_matrices" / "aa_composition_100pct.png"
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        fig = plot_confusion_matrix(y_test, full_predictions, "aa_composition (100% training data)", cm_path)
        plt.close(fig)
        print(f"Saved confusion matrix to {cm_path}")

    summary = results[results["representation"] == "aa_composition"].groupby("fraction")[
        ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    ].agg(["mean", "std"])
    print(f"\nTest split: {TEST_SPLIT}")
    print(f"Saved per-seed metrics to {args.output}")
    print("\nMean and standard deviation over seeds:")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
