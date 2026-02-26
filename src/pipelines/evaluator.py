import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import *

from tqdm import tqdm

RESULTS = Dict[str, Union[List[float], "RESULTS"]]

class Evaluator:
    def evaluate(self, generation_file: str) -> Dict[str, float]:
        assert generation_file.endswith(".jsonl") or generation_file.endswith(".json")
        assert os.path.exists(generation_file)
        
        generations = []
        if generation_file.endswith(".json"):
            with open(generation_file, "r", encoding="utf-8") as f:
                generations = json.load(f)
        elif generation_file.endswith(".jsonl"):
            with open(generation_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    generations.append(item)
        else:
            raise ValueError(f"Unsupported file format: {generation_file}")
            
        results: RESULTS = {}
        
        def merge_result(results: RESULTS, new_result: Dict[str, Any]) -> None:
            """
            Merge a new_result dict (which may have nested dicts) into the cumulative results.
            If the new value is a dict, merge it recursively.
            If the new value is a float, append it to the corresponding list.
            """
            for key, value in new_result.items():
                if isinstance(value, dict):
                    # Ensure that results[key] is a dict.
                    if key not in results or not isinstance(results[key], dict):
                        results[key] = {}
                    merge_result(results[key], value)
                else:
                    # Assume value is a float.
                    results.setdefault(key, []).append(value)

        def average_results(results: Union[List[float], RESULTS]) -> Union[float, Dict[str, float]]:
            """
            Recursively average all lists of floats. If the value is a list,
            return its average; if it is a dict, apply recursively.
            """
            if isinstance(results, list):
                return sum(results) / len(results)
            elif isinstance(results, dict):
                return {k: average_results(v) for k, v in results.items()}
        
        for idx, generation in tqdm(enumerate(generations), "Evaluating"):
            result = self.evaluate_item(idx, generation)
            # for key, value in result.items():
            #     results.setdefault(key, []).append(value)
            merge_result(results, result)
        
        return average_results(results)
    
    def evaluate_item(self, idx: int, generation) -> Dict[str, float]:
        raise NotImplementedError()