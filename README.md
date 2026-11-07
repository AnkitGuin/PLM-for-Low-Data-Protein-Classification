# Protein Language Models for Low-Data Protein Classification

This repository currently implements a reproducible low-data **amino-acid
composition baseline** on the TAPE Remote Homology dataset. It tests a simple
question: how does fold-classification performance change as the number of
labeled training proteins decreases?

ESM-2 and ProtT5 pipelines are not present in this checkout. Results
produced here therefore cover amino-acid composition only and do not support
a PLM-versus-baseline conclusion.

## Dataset and evaluation

The loader downloads the official TAPE raw JSON archive on its first run and
caches the data under `data/`. It uses the supplied train and validation
splits without reshuffling. By default, evaluation uses TAPE's
`test_fold_holdout` set. The task has 1,195 fold labels; the default test split
contains a subset of those labels. The loader and run report count classes
missing from each low-data training sample so that their impact is visible.

The [TAPE benchmark paper](https://papers.neurips.cc/paper/9163-evaluating-protein-transfer-learning-with-tape)
defines remote homology as classification over 1,195 protein folds and reports accuracy on the fold-level held-out set. This
project also reports macro F1, precision, and recall to show performance
across the unevenly represented test classes. These are additional project
metrics, not the headline metric used in the original benchmark.

## Setup and run

From this directory:

Use Python 3.12 or newer.

```bash
python -m pip install -r requirements.txt
python -m src.run_baseline
```

The first run needs internet access to download the dataset (about 120 MB).
Subsequent runs use the cached files. The default experiment uses fractions
`1%, 5%, 10%, 25%, 50%, 100%` and three seeds (`0`, `1`, `2`). Within each seed,
smaller samples are nested subsets of larger samples. No stratification is
used, so rare folds can be absent at low fractions; this is part of the
low-data setting and is recorded in the metrics.

Use custom seeds/fractions if desired:

```bash
python -m src.run_baseline --seeds 0 1 2 --fractions 0.01 0.1 1.0
```

The official test set is used only for final scoring; the current baseline
does not tune hyperparameters on the validation split. Logistic Regression
uses the same fixed settings throughout (`lbfgs`, `max_iter=2000`, default
regularization) and amino-acid composition is represented by 20 frequencies
in `ACDEFGHIKLMNPQRSTVWY` order.

## Outputs

- `results/metrics.csv`: one row per seed and training fraction, including
  accuracy, macro metrics, sample size, observed class counts, and the number
  of test classes absent from that training sample.
- `results/confusion_matrices/aa_composition_100pct.png`: confusion matrix
  for the 100% model from the first seed.
- `results/learning_curve.png`: mean and standard deviation by training fraction.
- `report.md`: the checked-in summary of the latest run; update it when you
  record a new experiment. Metrics and plots are included as run artifacts;
  downloaded data and PLM embeddings remain gitignored.

## Repository layout

```text
src/
  data.py          # download, safely extract, and load official TAPE splits
  features.py      # amino-acid composition baseline
  train.py         # reproducible nested sampling and Logistic Regression
  evaluate.py      # metrics and plots
  run_baseline.py  # end-to-end command-line experiment
notebooks/
  01_data_exploration.ipynb
  02_baseline.ipynb  # notebook entry point for the reproducible runner
```

## Scope and limitations

This is a baseline implementation, not a reproduction of TAPE's neural
models. It has one representation and one fixed classifier configuration;
there is no ESM-2 or ProtT5 comparison, hyperparameter search, or repeated
cross-validation. Three random seeds provide a basic measure of sampling
variability, but the reported standard deviation is not a confidence interval
and should not be interpreted as one.
