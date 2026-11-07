"""
features.py
------------
Baseline protein representation: amino-acid composition.

Each protein sequence is turned into a 20-dimensional vector of the relative
frequency of each of the 20 standard amino acids. This is a simple, fast,
non-learned representation used as the project's baseline.
"""

import numpy as np

# Fixed, alphabetical order so every feature vector uses the same column order.
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def amino_acid_composition(sequence):
    """
    Convert a single protein sequence into a 20-dim amino-acid frequency vector.

    Any character not in the 20 standard amino acids (e.g. 'X', 'B', 'U' for
    non-standard/ambiguous residues) is ignored when computing frequencies.
    """
    sequence = sequence.upper()
    counts = np.zeros(len(STANDARD_AMINO_ACIDS), dtype=np.float64)

    for i, aa in enumerate(STANDARD_AMINO_ACIDS):
        counts[i] = sequence.count(aa)

    n_standard = counts.sum()
    if n_standard == 0:
        return counts  # empty/all-ambiguous sequences have no composition

    # Normalize over canonical residues only; ambiguous symbols must not make
    # otherwise valid amino-acid frequencies sum to less than one.
    return counts / n_standard


def compute_composition_features(sequences):
    """Apply amino_acid_composition to a list; returns an (N, 20) array."""
    features = [amino_acid_composition(seq) for seq in sequences]
    if not features:
        return np.empty((0, len(STANDARD_AMINO_ACIDS)), dtype=np.float64)
    return np.stack(features)


if __name__ == "__main__":
    # Quick manual check: python src/features.py
    example = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
    vec = amino_acid_composition(example)
    print("vector length:", len(vec))
    print("sums to 1.0:", np.isclose(vec.sum(), 1.0))
    print(dict(zip(STANDARD_AMINO_ACIDS, np.round(vec, 3))))
