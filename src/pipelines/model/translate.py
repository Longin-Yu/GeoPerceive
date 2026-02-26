import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import *

from src.pipelines.model_api import handle_responses
from openai import OpenAI

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--src', type=str, required=True, help='Source file path'
    )
    arg_parser.add_argument(
        '--dst', type=str, required=True, help='Destination file path'
    )
    arg_parser.add_argument(
        '--src_column', type=str, default="prediction", help='Source column name'
    )
    arg_parser.add_argument(
        '--dst_column', type=str, default="translation", help='Destination column name'
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
        
    existing_src_ids = set()
    if os.path.exists(args.dst):
        with open(args.dst, 'r') as f:
            existing_lines = f.readlines()
            for line in existing_lines:
                line = json.loads(line)
                existing_src_ids.add(line['src_id'])
        print(f'\033[1;36m' + f'Excluding {len(existing_src_ids)} existing samples' + '\033[0m')
    
    queries = []
    raw_data = []
    for src_id, line in enumerate(lines):
        if src_id in existing_src_ids:
            continue
        raw_data.append({**line, 'src_id': src_id})
        queries.append(line[args.src_column])
    
    print('\033[1;36m' + f'Total Samples: {len(queries)}' + '\033[0m')
    
    def callback(index, response):
        if response is not None:
            with open(args.dst, 'a+') as f:
                f.write(json.dumps({**raw_data[index], args.dst_column: response}) + '\n')
    
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