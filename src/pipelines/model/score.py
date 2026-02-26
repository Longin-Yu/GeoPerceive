import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import *

from tqdm import tqdm
import fire

from src.utils import load_json_file, save_json_file
from src.pipelines.model_api import handle_responses
from src.prompts import CONSTRUCTION2NL_PROMPT
from src.definition import GeometryConstruction
from src.parser import *
from src.pipelines.evaluator import Evaluator
from src.evaluation import ConstructionMatching

def main(
    meta_path: str,
    translation_path: str,
    save_path: str,
    translation_column: str = "translation",
):
    meta = load_json_file(meta_path)
    translation = load_json_file(translation_path)
    
    print(f'\033[1;36mHandling Ground Truth.\033[0m')
    
    id2gt = {}
    for item in tqdm(meta):
        assert item['index'] not in id2gt, f"Duplicate index {item['index']} in meta data."
        gt = GeometryConstruction.from_json(item["data"])
        id2gt[item['index']] = gt
    
    print(f'\033[1;36mHandling Predictions.\033[0m')
    
    saves = []
    for item in tqdm(translation):
        parser = Parser()
        score = 0
        try:
            pred = parser.parse(item[translation_column])
            score = ConstructionMatching(pred, gt).scores
            score_overall = (
                score["points"]["f1"] 
                + score["lines"]["f1"] 
                + score["circles"]["f1"]
                + score["constraints"]["f1"]
            ) / 4
            saves.append({
                **item,
                "score": score_overall,
            })
        except ParserError:
            saves.append({
                **item,
                "score": 0,
            })
    
    save_json_file(saves, save_path)
    

if __name__ == '__main__':
    fire.Fire(main)