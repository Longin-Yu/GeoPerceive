#!/usr/bin/env python3
# balance_jsonl.py
"""
Balance a meta.jsonl file into a new balanced.jsonl file.

Usage example
-------------
python balance_jsonl.py \
  --src "../generated/v2-extra/meta.jsonl" \
  --dst "../generated/v2-extra/balanced.jsonl" \
  --cnt "[0,40000,26000,20000,9000,5000]" \
  --seed 123
"""

import json
import random
from pathlib import Path
from typing import List

import fire


def balance(
    src: str,
    dst: str,
    cnt: List[int] = (0, 40_000, 26_000, 20_000, 9_000, 5_000),
    seed: int = 42,
) -> None:
    """Create a balanced subset of a meta.jsonl file.

    Parameters
    ----------
    src : str
        Path to the input JSONL file.
    dst : str
        Path to the output JSONL file.
    cnt : List[int], default = (0,40000,26000,20000,9000,5000)
        Number of samples to keep for steps 0-5.
        Provide either a Python-style list string ('[0,1,2]') or a
        comma-separated string ('0,1,2').
    seed : int, default = 42
        RNG seed for `random.shuffle` to ensure reproducibility.
    """

    # --- Normalise and sanity-check arguments --------------------------------
    if isinstance(cnt, str):
        cnt = (
            list(map(int, cnt.strip("[]() ").split(",")))
            if "," in cnt
            else [int(cnt)]
        )
    if len(cnt) < 6:
        raise ValueError("`cnt` must have at least six integers (for steps 0-5).")

    src_path, dst_path = Path(src), Path(dst)
    if not src_path.is_file():
        raise FileNotFoundError(src)

    # --- Read and bucket data by step ----------------------------------------
    buckets: List[List[dict]] = [[] for _ in range(6)]
    with src_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            step = item.get("step")
            if step is None or not (0 <= step < 6):
                continue  # skip malformed or out-of-range entries
            buckets[step].append(item)

    # --- Shuffle & truncate each bucket --------------------------------------
    random.seed(seed)
    selected: List[dict] = []
    for step in range(1, 6):  # step 0 is ignored per original script
        random.shuffle(buckets[step])
        selected.extend(buckets[step][: cnt[step]])

    # --- Write balanced file --------------------------------------------------
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"✅  Wrote {len(selected):,} lines to {dst_path} "
        f"(seed={seed}, counts={cnt[1:6]})"
    )


if __name__ == "__main__":
    # Expose the `balance` function as a Fire CLI
    fire.Fire(balance)
