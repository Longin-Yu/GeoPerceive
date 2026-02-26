import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import *

import fire
from tqdm import tqdm

from src.pipelines.model_api import handle_responses
from src.definition import GeometryConstruction
from src.parser import *

def load_file(file_path: str):
    if file_path.endswith(".jsonl"):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines]
    elif file_path.endswith(".json"):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def prepare_llama_factory_format(
    meta_file_path: str,
    caption_file_path: str,
    caption_column: str,
    save_file_path: str,
):
    metas = load_file(meta_file_path)
    id2meta = {item['index']: item for item in metas}
    captions = load_file(caption_file_path)
    
    saves = []
    for caption in tqdm(captions):
        index = caption['index']
        construction = GeometryConstruction.from_json(id2meta[index]["data"])
        code = dump_to_code(construction)
        nl = caption[caption_column]
        saves.append({
            "index": index,
            "instruction": nl,
            "input": "",
            "output": code
        })
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    with open(save_file_path, 'w') as f:
        json.dump(saves, f, indent=4)

def main():
    fire.Fire(prepare_llama_factory_format)

if __name__ == '__main__':
    main()