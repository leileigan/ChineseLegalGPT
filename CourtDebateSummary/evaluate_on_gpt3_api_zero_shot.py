import argparse
import datetime
import os
import pickle
import random
import sys
import time
import json
import re
import logging

import numpy as np
import torch
from tqdm import tqdm
import tiktoken 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from typing import List, Tuple

import openai
from rouge import Rouge


TASK_DESCRIPTION = [
    """请为以下庭审对话撰写一份摘要，要求摘要能够保留对话的重要信息。其中“审”代表法官，“原”代表原告，“被”代表被告。"""
    ]

ANSWER_TEMPLATE = ["庭审对话:\n{dialogue}\n请为上述对话的撰写一份摘要："
    ]


class Request(object):
    def __init__(self, engine, temperature, max_tokens, top_p, n, log_probs, frequency_penalty, presence_penalty, best_of):
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.n = n
        self.log_probs = log_probs
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.best_of = best_of

    def get_multiple_sample(self, prompt_list: List[str]):
    
        while True:
            try:
                response = openai.Completion.create(
                    engine=self.engine,
                    prompt=prompt_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    n = self.n,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    best_of=self.best_of
                )
                return response
            except Exception as err:
                logging.exception(err)
                time.sleep(60)


def seed_rand(SEED_NUM):
    torch.random.manual_seed(SEED_NUM)
    torch.manual_seed(SEED_NUM)
    random.seed(SEED_NUM)
    np.random.seed(SEED_NUM)
    torch.cuda.manual_seed(SEED_NUM)
    torch.cuda.manual_seed_all(SEED_NUM)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

def read_datasets(path):
    fp = open(path, 'r')
    raw_data = json.load(fp)
    datasets = []
    for k, v in raw_data.items():
        datasets.append(("".join(v["trial_text"].split()), v["factfinding_text"]))
    
    return datasets


def read_dump_file(dump_path):
    fr = open(dump_path, 'r')
    golden_list, pred_list = [], []

    for l in fr.readlines():
        try:
            d = json.loads(l)
        except:
            print("cannot parse line:", l)

        if "input" not in d.keys():
            continue
        golden_list.append(d['golden'])
        pred_list.append(d['pred'])
    fr.close()
    return golden_list, pred_list


def metrics(golden_list, pred_list):
    
    rouge = Rouge()
    rouge_1, rouge_2, rouge_l = 0, 0, 0 

    for pred, ref in tqdm(zip(pred_list, golden_list)):
        scores = rouge.get_scores(pred, ref)
        rouge_1 += scores[0]['rouge-1']['f']
        rouge_2 += scores[0]['rouge-2']['f']
        rouge_l += scores[0]['rouge-l']['f']

    total = len(golden_list)
    rouge_1 = rouge_1/total
    rouge_2 = rouge_2/total
    rouge_l = rouge_l/total

    scores = {"rouge1": rouge_1, "rouge2": rouge_2, "rougel": rouge_l}

    return scores

def post_processing_response(resp, input_list, sample_id_list, golden_label_list, wf):

    for resp_id in range(len(input_list)):
        resp_text = resp['choices'][0]['text'].strip()
        tmp = {"sample_id": sample_id_list[resp_id], "input": input_list[resp_id], "response": resp['choices'][0], "pred":resp_text , "golden": golden_label_list[resp_id], "model": resp['model']}
        wf.write(json.dumps(tmp) + "\n")


def zero_shot_evaluate_on_llm(request: Request, testdataset: List[Tuple[str, str]], start: int, end: int, dump_path: str, task_description_id: int, example_template_id: int):
    """zero shot evaluation on large languge models (llm)

    Args:
        request (Request): api request to llm
        testdataset (List[Tuple[str, str]]): test dataset to be evaluated
        start (int): start index of the test dataset
        end (int): end index of the test dataset
        dump_path (str): dump path for the api responses
        task_description_id (int): decide which task description is used
        example_template_id (int): decide which example template is used
    """
    prompts = [TASK_DESCRIPTION[task_description_id]]
    if os.path.exists(dump_path):
        print(f"File path {dump_path} exists")
        wf = open(dump_path, 'a')
    else:
        print(f"Create file {dump_path}")
        wf = open(dump_path, 'w+')

    input_list, sample_id_list, golden_label_list = [], [], []
    for sample_id in range(start, end):
        dialogue, summary = testdataset[sample_id]
        template = ANSWER_TEMPLATE[example_template_id]
        test_example = template.format(dialogue=dialogue)
        input = f"{prompts}\n{test_example}"
        input_list.append(input)
        sample_id_list.append(sample_id)
        golden_label_list.append(summary)
        if len(input_list) == 5:
            resp = request.get_multiple_sample(input_list)
            post_processing_response(resp, input_list, sample_id_list, golden_label_list, wf)
            time.sleep(5)
            print("#", end="", flush=True)
            input_list, sample_id_list, golden_label_list = [], [], []

    if len(input_list):
        resp = request.get_multiple_sample(input_list)
        post_processing_response(resp, input_list, sample_id_list, golden_label_list, wf)

    wf.close()

    golden_list, pred_list = read_dump_file(dump_path)
    score = metrics(golden_list, pred_list)

    wf = open(dump_path, 'a')
    wf.write(json.dumps(score)+"\n")


if __name__ == '__main__':
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser(description='Chinese Legal GPT')
    parser.add_argument('--testdata_path', required=True, type=str)
    parser.add_argument('--dump_path', default='', type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=1000, type=int)
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--task_description_id', type=int, default=2)
    parser.add_argument('--example_template_id', type=int, default=1)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--logprobs', default=5)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--max_tokens', type=int, default=200)

    args = parser.parse_args()
    print(args)

    openai.api_key = args.api_key
    testdata_path = args.testdata_path
    dump_path = args.dump_path
    start = args.start
    end = args.end
    model_name = args.model_name
    task_description_id = args.task_description_id
    example_template_id = args.example_template_id
    temperature = args.temperature
    logprobs = args.logprobs
    max_tokens = args.max_tokens

    request = Request(
        engine=model_name,  
        temperature=temperature,  
        max_tokens=max_tokens,  
        top_p=1,  
        frequency_penalty=0,  
        presence_penalty=0,  
        log_probs=logprobs
    )

    test_datasets = read_datasets(testdata_path)
    zero_shot_evaluate_on_llm(request, test_datasets, start, end, dump_path, task_description_id, example_template_id)