import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import *

from copy import deepcopy
from tqdm import tqdm
import fire

from src.utils import load_json_file, save_json_file
from src.prompts import CAPTION_SAMPLE_PROMPT

def main(
    meta_path: str,
    score_path: str,
    save_path: str,
    pair_score_threshold: float = 0.3,
):
    
    metas = load_json_file(meta_path)
    scores = load_json_file(score_path)
    
    print('\033[1;36m' + f'Found {len(scores)} samples.' + '\033[0m')
    
    id2data = {}
    for meta in metas:
        id2data[meta["index"]] = {
            "meta": meta,
            "samples": [],
        }
    
    for score in scores:
        data = id2data[score["index"]]
        data["samples"].append({
            "sample": score["prediction"],
            "score": score["score"],
            "src_id": score["src_id"],
        })
    
    saves = []
    for index, value in tqdm(id2data.items()):
        image_path = os.path.abspath(os.path.join(os.path.dirname(meta_path), "images", f"{index}.jpg"))
        TEMPLATE = {
            "index": index,
            "instruction": "<image>\n" + CAPTION_SAMPLE_PROMPT,
            "input": "",
            "images": [image_path]
        }
        samples = sorted(value["samples"], key=lambda x: x["score"], reverse=True)
        chosen_idx = 0
        reject_idx = (len(samples) + 1) // 2
        while chosen_idx < len(samples) // 2 and reject_idx < len(samples):
            delta = samples[chosen_idx]["score"] - samples[reject_idx]["score"]
            if delta < pair_score_threshold:
                reject_idx += 1
                continue
            save = deepcopy(TEMPLATE)
            save.update({
                "chosen": samples[chosen_idx]["sample"],
                "rejected": samples[reject_idx]["sample"],
                "chosen_score": samples[chosen_idx]["score"],
                "rejected_score": samples[reject_idx]["score"],
                "delta_score": delta,
                "chosen_src_id": samples[chosen_idx]["src_id"],
                "rejected_src_id": samples[reject_idx]["src_id"],
            })
            saves.append(save)
            chosen_idx += 1
            reject_idx += 1

    print('\033[1;36m' + f'Got {len(saves)} pairs.' + '\033[0m')
    
    save_json_file(saves, save_path)

if __name__ == '__main__':
    fire.Fire(main)