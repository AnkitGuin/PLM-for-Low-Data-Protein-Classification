"""
train.py
--------
Two small building blocks used by the baseline's low-data experiments:

1. sample_fraction  -- take a reproducible random subset of the training set
2. train_logistic_regression -- fit the (single, shared) downstream classifier

Keeping sampling and classifier fitting separate from feature construction
makes the baseline procedure easy to reproduce and extend.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def sample_fraction(X, y, fraction, seed):
    """
    Randomly select `fraction` of the training examples (e.g. 0.05 for 5%).

    A fixed seed makes the subsample reproducible. We use plain random
    subsampling (not stratified) to keep this simple -- at very small
    fractions (e.g. 1%) some rare classes may end up missing entirely,
    which is a realistic and expected feature of low-data settings.
    """
    n_total = len(y)
    if len(X) != n_total:
        raise ValueError(f"X and y must have the same number of rows, got {len(X)} and {n_total}")
    if n_total == 0:
        raise ValueError("Cannot sample from an empty training set")
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be greater than 0 and at most 1, got {fraction}")
    n_sample = max(1, int(round(n_total * fraction)))

    # A seeded permutation makes the subsets nested as `fraction` grows.
    # Independent choice calls can otherwise produce unrelated samples and
    # add avoidable noise to the learning curve.
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)[:n_sample]

    X_sampled = X[indices]
    y_sampled = np.asarray(y)[indices]
    return X_sampled, y_sampled


def train_logistic_regression(X_train, y_train, seed, max_iter=2000):
    """
    Fit a multi-class Logistic Regression classifier.

    This is the baseline classifier for the amino-acid composition features.
    """
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=seed,
        # 'lbfgs' handles multi-class problems natively (multinomial loss)
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    # Quick manual check with synthetic data: python src/train.py
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_classes=4, random_state=0
    )

    X_sub, y_sub = sample_fraction(X, y, fraction=0.1, seed=42)
    print(f"sampled {len(y_sub)} / {len(y)} examples")

    model = train_logistic_regression(X_sub, y_sub, seed=42)
    train_acc = model.score(X_sub, y_sub)
    print(f"train accuracy on the 10% subset: {train_acc:.3f}")
