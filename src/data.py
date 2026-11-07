"""
data.py
-------
Downloads and loads the official TAPE Remote Homology dataset splits.

Source: the TAPE benchmark (https://github.com/songlab-cal/tape), Rao et al.
2019. We use the "raw" JSON release of the data rather than the LMDB release,
since it doesn't require the `lmdb` package and is trivial to parse with the
standard library.

Official download URL (public, no login/API key needed):
    https://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/remote_homology.tar.gz

We never reshuffle the official train/validation/test splits. TAPE supplies
three held-out test variants, each withholding a different evolutionary
grouping level:
    test_fold_holdout
    test_superfamily_holdout
    test_family_holdout
Change TEST_SPLIT below if you want to evaluate on a different one.
"""

import json
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlopen

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARCHIVE_URL = "https://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/remote_homology.tar.gz"
TEST_SPLIT = "test_fold_holdout"

VALID_SPLITS = [
    "train", "valid",
    "test_fold_holdout", "test_family_holdout", "test_superfamily_holdout",
]

# Different releases of the raw TAPE data have used slightly different key
# names for the same fields, so we try a short list of candidates instead of
# hard-coding one -- this keeps data.py from breaking silently if a key name
# differs from what we expect.
SEQUENCE_KEY_CANDIDATES = ["primary", "seq", "sequence"]
LABEL_KEY_CANDIDATES = ["fold_label", "label", "class_label"]


def _download_and_extract(data_dir: Path = DATA_DIR) -> Path:
    """Download + untar the official archive if it isn't already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = data_dir / "remote_homology"

    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        return extracted_dir

    archive_path = data_dir / "remote_homology.tar.gz"
    print(f"Downloading TAPE Remote Homology data from {ARCHIVE_URL} ...")
    try:
        with urlopen(ARCHIVE_URL, timeout=60) as response, archive_path.open("wb") as out:
            shutil.copyfileobj(response, out)

        print("Extracting...")
        root = data_dir.resolve()
        with tarfile.open(archive_path, "r:gz") as tar:
            # Reject links and paths outside the destination before extracting
            # an archive fetched from the network.
            members = tar.getmembers()
            for member in members:
                target = (data_dir / member.name).resolve()
                if target != root and root not in target.parents:
                    raise ValueError(f"Unsafe path in dataset archive: {member.name}")
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsupported member in dataset archive: {member.name}")
            tar.extractall(data_dir, members=members, filter="data")
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    else:
        archive_path.unlink()  # remove the .tar.gz once extracted

    return extracted_dir


def _find_split_file(extracted_dir: Path, split: str) -> Path:
    """Locate the JSON file for one split, tolerating minor naming differences."""
    expected = extracted_dir / f"remote_homology_{split}.json"
    if expected.exists():
        return expected

    matches = list(extracted_dir.rglob(f"*{split}*.json"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find a data file for split '{split}' in {extracted_dir}. "
        f"Contents found: {list(extracted_dir.rglob('*'))[:20]}"
    )


def _pick_key(record: dict, candidates, kind: str) -> str:
    for key in candidates:
        if key in record:
            return key
    raise KeyError(
        f"Could not find a {kind} field. Looked for {candidates}, but a "
        f"record only has keys: {list(record.keys())}. Inspect the data in "
        f"01_data_exploration.ipynb and update the candidate list in data.py."
    )


def load_split(split: str, data_dir: Path = DATA_DIR):
    """Load one split as (sequences, labels) -- two plain Python lists."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")

    extracted_dir = _download_and_extract(data_dir)
    split_file = _find_split_file(extracted_dir, split)

    with open(split_file, encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        raise ValueError(f"Split '{split}' is empty: {split_file}")

    seq_key = _pick_key(records[0], SEQUENCE_KEY_CANDIDATES, "sequence")
    label_key = _pick_key(records[0], LABEL_KEY_CANDIDATES, "label")

    missing = [i for i, record in enumerate(records) if seq_key not in record or label_key not in record]
    if missing:
        raise ValueError(f"Split '{split}' is missing expected fields in records, first indices: {missing[:10]}")

    sequences = [r[seq_key] for r in records]
    labels = [r[label_key] for r in records]
    return sequences, labels


def load_train_val_test(test_split: str = TEST_SPLIT, data_dir: Path = DATA_DIR):
    """
    Convenience loader for the three splits used throughout the project.

    Returns
    -------
    (train_x, train_y), (val_x, val_y), (test_x, test_y)
    """
    train = load_split("train", data_dir)
    val = load_split("valid", data_dir)
    test = load_split(test_split, data_dir)
    return train, val, test


if __name__ == "__main__":
    # Quick manual check: python src/data.py
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_train_val_test()
    print(f"train: {len(train_x)} sequences, {len(set(train_y))} unique labels")
    print(f"valid: {len(val_x)} sequences")
    print(f"test:  {len(test_x)} sequences")
    print("example sequence:", train_x[0][:50], "...")
    print("example label:", train_y[0])
