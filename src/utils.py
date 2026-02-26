import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import *

def load_json_file(file_path: str) -> List:
    if file_path.endswith(".jsonl"):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines]
    elif file_path.endswith(".json"):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def save_json_file(data: list, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if file_path.endswith(".jsonl"):
        with open(file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    elif file_path.endswith(".json"):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")