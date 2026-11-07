# Run report: TAPE Remote Homology composition baseline

**Run date:** 2026-11-07  
**Configuration:** amino-acid composition + scikit-learn Logistic Regression, evaluated on `test_fold_holdout`  
**Reproduction command:** `python -m src.run_baseline`

## Dataset and method

The official TAPE raw JSON archive loaded successfully. The training split has
12,312 sequences across 1,195 fold labels. The selected test split has 718
sequences across 136 labels; all 136 labels occur in the full training split.
Test class sizes range from 1 to 30 sequences (median 4), so accuracy alone
does not describe performance across folds.

Each protein is represented by 20 amino-acid frequencies normalized over canonical residues. The official train and test sequences contained no ambiguous residues, so this normalization choice does not alter the reported run. A multinomial
Logistic Regression classifier (`lbfgs`, `max_iter=2000`, default `C=1`) was
trained on nested random subsets at six fractions. Results use three sampling
seeds (0, 1, 2). The test set was used only for final scoring. Macro metrics
are averaged over the fixed 136 labels present in the test set; an absent
training class therefore receives zero recall/F1 when it appears in test.

## Results

Values are mean ± sample standard deviation across the three seeds. Scores
are fractions (for example, `0.0724` means 7.24%).

| Training data | Examples | Accuracy | Macro F1 | Macro precision | Macro recall |
|---:|---:|---:|---:|---:|---:|
| 1% | 123 | 0.0265 ± 0.0024 | 0.0006 ± 0.0003 | 0.0003 ± 0.0002 | 0.0070 ± 0.0006 |
| 5% | 616 | 0.0279 ± 0.0000 | 0.0004 ± 0.0000 | 0.0002 ± 0.0000 | 0.0074 ± 0.0000 |
| 10% | 1,231 | 0.0279 ± 0.0000 | 0.0004 ± 0.0000 | 0.0002 ± 0.0000 | 0.0074 ± 0.0000 |
| 25% | 3,078 | 0.0362 ± 0.0042 | 0.0045 ± 0.0015 | 0.0066 ± 0.0002 | 0.0104 ± 0.0014 |
| 50% | 6,156 | 0.0529 ± 0.0037 | 0.0077 ± 0.0006 | 0.0078 ± 0.0008 | 0.0158 ± 0.0009 |
| 100% | 12,312 | 0.0724 ± 0.0000 | 0.0092 ± 0.0000 | 0.0080 ± 0.0000 | 0.0196 ± 0.0000 |

The majority test class contains 30/718 sequences, giving a majority-class
accuracy of 4.18%. The full-data composition model reaches 7.24% accuracy
(3.06 percentage points above that reference), but its macro F1 is only
0.92%. At 1% training data, only 82–86 of the 1,195 labels appear in the
sample, and 98–107 of the 136 test labels are absent from that sample. At
100%, all 1,195 training labels are observed. All 18 fits converged before
the 2,000-iteration limit (maximum observed iteration count: 64).

## Conclusion and scope

**Resume-ready summary:** Implemented and ran a reproducible, three-seed low-data protein fold-classification baseline on TAPE using 20-dimensional amino-acid composition and multinomial Logistic Regression; full-data accuracy was 7.24% versus 4.18% majority-class accuracy.


More labeled data improved accuracy in this run, from 2.65% at 1% to 7.24%
at 100%. The very low macro F1 shows that amino-acid composition plus a linear
classifier does not reliably distinguish the test folds, especially when
many labels are missing from low-data subsets. This establishes a baseline
and supports evaluating richer sequence representations next.

This checkout does **not** contain ESM-2 or ProtT5 extraction/classification
pipelines, so these results cannot support a claim that a protein language
model outperforms composition. Three seeds give only a preliminary estimate
of sampling variability; zero standard deviation at 100% reflects identical
training examples and a deterministic fit, not evidence of general
statistical certainty. The original TAPE benchmark reports top-1 accuracy;
macro metrics here are additional project diagnostics.

## Artifacts and environment

- Per-seed metrics: `results/metrics.csv`
- Full-data confusion matrix (seed 0): `results/confusion_matrices/aa_composition_100pct.png`
- Python 3.13.3; NumPy 2.2.5; pandas 2.2.3; scikit-learn 1.6.1; Matplotlib 3.10.1

Reference: [TAPE paper](https://papers.neurips.cc/paper/9163-evaluating-protein-transfer-learning-with-tape) and [official TAPE repository](https://github.com/songlab-cal/tape).
