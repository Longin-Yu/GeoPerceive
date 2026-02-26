import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import *

from src.pipelines.model_api import handle_responses
from src.prompts import CAPTION_SAMPLE_PROMPT
from openai import OpenAI
import base64

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--src', type=str, required=True, help='Source file path'
    )
    arg_parser.add_argument(
        '--dst', type=str, required=True, help='Destination file path'
    )
    arg_parser.add_argument(
        '--num_samples', type=int, default=10, help='Number of samples for each piece of data to generate'
    )
    arg_parser.add_argument(
        '--client', type=str, nargs='+', required=True, help="Client URLs"
    )
    return arg_parser.parse_args()



def main():
    args = parse_args()
    
    if args.src.endswith('.json'):
        with open(args.src, 'r') as f:
            lines = json.load(f)
    else:
        with open(args.src, 'r') as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
    idx2data: Dict[int, dict] = {
        line['index']: line for line in lines
    }
    idx2cnt: Dict[int, int] = {
        line['index']: args.num_samples for line in lines
    }
    
    if os.path.exists(args.dst):
        print('\033[1;36m' + 'Output file exists. Excluding existing sampled data.' + '\033[0m')
        with open(args.dst, 'r') as f:
            existing_lines = f.readlines()
        for line in existing_lines:
            line = json.loads(line)
            idx2cnt[line['index']] -= 1
            if idx2cnt[line['index']] == 0:
                del idx2cnt[line['index']]
    
    queries = []
    indices = []
    for key, cnt in idx2cnt.items():
        image_path = os.path.join(os.path.dirname(args.src), "images", f"{key}.jpg")
        with open(image_path, 'rb') as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        query = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            {"type": "text", "text": CAPTION_SAMPLE_PROMPT}
        ]
        queries.extend([query] * cnt)
        indices.extend([key] * cnt)
        
    # with open("local_debug.json", "w") as f:
    #     json.dump(queries[:10], f, indent=4)
    #     exit(0)
    
    print('\033[1;36m' + f'Total Samples: {len(queries)}' + '\033[0m')
    
    def callback(index, response):
        if response is not None:
            with open(args.dst, 'a+') as f:
                f.write(json.dumps({"index": indices[index], "prediction": response}) + '\n')
    
    handle_responses(
        "", queries=queries, processes=len(args.client)*3, callback=callback,
        client=[
            OpenAI(
                api_key="0",
                base_url=client
            ) for client in args.client
        ]
    )
    

if __name__ == '__main__':
    main()