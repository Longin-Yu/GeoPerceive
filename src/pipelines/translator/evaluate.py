import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import *

from tqdm import tqdm
import fire

from src.pipelines.model_api import handle_responses
from src.prompts import CONSTRUCTION2NL_PROMPT
from src.definition import GeometryConstruction
from src.parser import *
from src.pipelines.evaluator import Evaluator
from src.evaluation import ConstructionMatching
from src.utils import load_json_file, save_json_file

class TranslatorEvaluator(Evaluator):
    def __init__(self, dataset_path: str, meta_path: str, generation_column: str = "predict", generation_index_column: Optional[str] = None):
        self.dataset = load_json_file(dataset_path)
        self.meta = load_json_file(meta_path)
        self.id2meta: Dict[str, dict] = dict()
        self.generation_column = generation_column
        self.generation_index_column = generation_index_column
        for item in self.meta:
            if 'index' not in item:
                raise ValueError(f"Meta item {item} does not contain 'index'.")
            self.id2meta[item['index']] = item
        
    def evaluate_item(self, idx, generation):
        real_idx = self.dataset[idx]['index'] if self.generation_index_column is None else generation[self.generation_index_column]
        gt_meta = self.id2meta[real_idx]
        steps = gt_meta["step"]
        return_wrap = lambda x: {"avg": x, "details": {f"{steps}": x}}
        META_TEMPLATE = {
            "validity": 1,
        }
        
        # pred_code = extract_code_block(generation["predict"])
        pred_code = generation[self.generation_column]
        if not pred_code:
            META_TEMPLATE.update({"validity": 0})
            return return_wrap(META_TEMPLATE)
        
        parser = Parser()
        try:
            parser.parse(pred_code.strip())
        except ParserError:
            # import traceback
            # traceback.print_exc()
            # exit()
            META_TEMPLATE.update({"validity": 0})
            return return_wrap(META_TEMPLATE)
        
        pred = GeometryConstruction(
            points=parser.points,
            lines=parser.lines,
            circles=parser.circles,
            constraints=parser.constraints,
        )
        
        gt = GeometryConstruction.from_json(gt_meta["data"])
        
        score = ConstructionMatching(pred, gt).scores
        META_TEMPLATE.update({
            "matching_score": score,
        })
        return return_wrap(META_TEMPLATE)

def main(
    meta_path: str,
    test_path: str,
    generation_path: str,
    save_path: str,
    generation_column: str = "predict",
    generation_index_column: Optional[str] = None,
):
    evaluator = TranslatorEvaluator(test_path, meta_path, generation_column, generation_index_column)
    results = evaluator.evaluate(generation_path)
    save_json_file(results, save_path)

if __name__ == '__main__':
    # main(
    #     meta_path = "/home/mnt/yuhao/workspace/projects/geometry/workspace/pytorch-geo-solver/generated/v2-construction2nl/meta.jsonl",
    #     test_path = "/home/mnt/yuhao/workspace/projects/geometry/workspace/pytorch-geo-solver/generated/v2-translator/test.json",
    #     generation_path = "/home/mnt/yuhao/workspace/projects/LLaMA-Factory/saves/qwen_2_5-7b/lora/sft-2/checkpoint-885-merged/generated_predictions.jsonl",
    #     save_path = "/home/mnt/yuhao/workspace/projects/geometry/workspace/pytorch-geo-solver/generated/v2-translator/test-results.json",
    # )
    fire.Fire(main)