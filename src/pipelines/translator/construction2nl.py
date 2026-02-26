#!/usr/bin/env python3
# construction2nl.py
"""
Generate natural-language (NL) descriptions from geometry-construction
sequences and append them to a JSONL file.
"""

import json
import os
from pathlib import Path
from typing import List

import fire
from openai import OpenAI

from src.pipelines.model_api import handle_responses
from src.prompts import CONSTRUCTION2NL_PROMPT_2


def construction2nl(
    src: str,
    dst: str,
    model: str = "Qwen/Qwen2.5-VL-72B-Instruct",
    processes: int = 10,
    prompt_template: str = CONSTRUCTION2NL_PROMPT_2,
    **openai_kwargs,
) -> None:
    """
    Parameters
    ----------
    src, dst : str
        Path to source JSONL (must contain `index` and `constructions`)
        and destination JSONL, respectively.
    model : str
        Model name passed to `handle_responses`.
    processes : int
        Worker processes for `handle_responses`.
    prompt_template : str
        f-string containing `{construction}` placeholder.
    openai_kwargs : dict | None
        Extra kwargs forwarded to `OpenAI(**openai_kwargs)`.
    """
    openai_kwargs = openai_kwargs or {}
    client = OpenAI(**openai_kwargs)

    # ------------------------------------------------------------------ load
    with Path(src).open("r", encoding="utf-8") as f:
        lines: List[dict] = [json.loads(line) for line in f]

    # ------------------------------------------------ skip already completed
    existing_ids: set[int] = set()
    if Path(dst).exists():
        with Path(dst).open("r", encoding="utf-8") as f:
            existing = [json.loads(line) for line in f]
        existing_ids = {item["index"] for item in existing}
        print(f"Found {len(existing_ids):,} existing NL entries → skipping.")

    lines = [ln for ln in lines if ln["index"] not in existing_ids]
    if not lines:
        print("Nothing to do. All entries already processed.")
        return

    queries = [
        prompt_template.format(construction=" ".join(ln["constructions"]))
        for ln in lines
    ]

    # ---------------------------------------------------------- callback
    def _callback(i: int, response: str | None) -> None:
        if response is None:
            return
        payload = {**lines[i], "nl": response}
        with Path(dst).open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------- run model
    handle_responses(
        model_name=model,
        queries=queries,
        processes=processes,
        callback=_callback,
        client=client,
    )

    print(f"✅  Appended {len(lines):,} NL entries to {dst}")


if __name__ == "__main__":
    fire.Fire(construction2nl)
