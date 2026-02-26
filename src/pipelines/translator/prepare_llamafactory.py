#!/usr/bin/env python3
# prepare_llamafactory.py
"""
Convert NL-annotated records to the JSON format expected by LLaMA-Factory,
and split into train / test files.
"""

import json
from pathlib import Path
from typing import List

import fire
from tqdm import tqdm

from src.definition import GeometryConstruction
from src.parser import dump_to_code


def prepare(
    src: str,
    out_dir: str,
    num_test: int = 500,
) -> None:
    """
    Parameters
    ----------
    src : str
        JSONL produced by `construction2nl.py` (must contain `data` and `nl`).
    out_dir : str
        Directory where `train.json` and `test.json` will be written.
    num_test : int
        Number of examples for the test split.
    """
    with Path(src).open("r", encoding="utf-8") as f:
        raw_lines: List[dict] = [json.loads(line) for line in f]

    records: list[dict] = []
    for ln in tqdm(raw_lines, desc="Converting"):
        gc = GeometryConstruction.from_json(ln["data"])
        code = dump_to_code(gc)
        records.append(
            {
                "index": ln["index"],
                "instruction": ln["nl"],
                "input": "",
                "output": code,
            }
        )

    if num_test > len(records):
        raise ValueError("`num_test` exceeds total record count.")

    train = records[:-num_test]
    test = records[-num_test:]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    (out_path / "train.json").write_text(
        json.dumps(train, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    (out_path / "test.json").write_text(
        json.dumps(test, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"✅  Saved {len(train):,} train and {len(test):,} test samples "
        f"to {out_path.relative_to(Path.cwd())}"
    )


if __name__ == "__main__":
    fire.Fire(prepare)
