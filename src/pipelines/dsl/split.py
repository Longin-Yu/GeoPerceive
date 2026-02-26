#!/usr/bin/env python3
# filter_dataset.py
"""
Filter a meta.jsonl file, then split the kept items into train / test sets.
"""

import json
import os
import random
from pathlib import Path
from typing import List

import fire


def main(
    data_dir: str,
    image_dir: str | None = None,
    meta_filename: str = "meta.jsonl",
    filtered_filename: str = "filtered.jsonl",
    train_filename: str = "train.jsonl",
    test_filename: str = "test.jsonl",
    loss_threshold: float = 1e-5,
    test_count: int = 200,
    seed: int = 42,
) -> None:
    """
    Parameters
    ----------
    data_dir : str
        Root directory that contains `meta.jsonl`.
    image_dir : str, optional
        Directory that holds per-sample record files (`<index>.json`).
        Defaults to `<data_dir>/images`.
    meta_filename, filtered_filename, train_filename, test_filename : str
        Filenames (not full paths) for the corresponding JSONL files.
    loss_threshold : float
        Keep samples whose loss is strictly below this value.
    test_count : int
        Number of examples to reserve for the test set (taken from the end of
        a random shuffle).
    seed : int
        RNG seed used before shuffling, so splits are reproducible.
    """
    # ------------------------------------------------------------------ paths
    data_path = Path(data_dir)
    if image_dir is None:
        image_path = data_path / "images"
    else:
        image_path = Path(image_dir)

    meta_path = data_path / meta_filename
    filtered_path = data_path / filtered_filename
    train_path = data_path / train_filename
    test_path = data_path / test_filename

    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)

    if not image_path.is_dir():
        raise FileNotFoundError(image_path)

    # -------------------------------------------------------------- load meta
    with meta_path.open("r", encoding="utf-8") as f:
        meta: List[dict] = [json.loads(line) for line in f]

    # --------------------------------------------------- filter by constraints
    filtered: List[dict] = []
    for item in meta:
        record_file = image_path / f"{item['index']}.json"
        if not record_file.is_file():
            # Skip if the per-sample file is missing
            continue
        with record_file.open("r", encoding="utf-8") as rf:
            record = json.load(rf)
        if record["constraints"]["loss"] < loss_threshold:
            filtered.append(item)

    # ---------------------------------------------------------- persist files
    filtered_path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in filtered) + "\n",
        encoding="utf-8",
    )

    # ------------------------------- shuffle & split into train / test sets
    rng = random.Random(seed)
    rng.shuffle(filtered)

    if test_count > len(filtered):
        raise ValueError(
            f"Requested {test_count} test samples, "
            f"but only {len(filtered)} items passed the filter."
        )

    train, test = filtered[:-test_count], filtered[-test_count:]

    train_path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in train) + "\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in test) + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------- summary
    print(
        f"✅  Kept {len(filtered):,} / {len(meta):,} samples "
        f"(loss < {loss_threshold}).\n"
        f"    Train: {len(train):,} → {train_path.relative_to(data_path)}\n"
        f"    Test : {len(test):,}  → {test_path.relative_to(data_path)}\n"
        f"    Filtered list saved to {filtered_path.relative_to(data_path)}"
    )


if __name__ == "__main__":
    fire.Fire(main)
