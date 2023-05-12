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
import math


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from typing import List

import openai
import torch
from rouge import Rouge


TASK_DESCRIPTION = [
    """请为以下庭审对话撰写一份摘要，要求摘要能够保留对话的重要信息。其中“审”代表法官，“原”代表原告，“被”代表被告。"""
    ]

EXAMPLE_TEMPLATE = ["庭审对话：\n{dialogue}\n对话摘要：\n{summary}"]

ANSWER_TEMPLATE = ["庭审对话：\n{dialogue}\n请为上述对话的撰写一份摘要："]


class Request(object):
    def __init__(self, api_key, engine, temperature, max_tokens, top_p, n, frequency_penalty, presence_penalty, best_of, logprobs=0):
        self.api_key = api_key
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.n = n
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.best_of = best_of
        self.logprobs = logprobs

    def __str__(self) -> str:
        return f"api: {self.api_key}, engine: {self.engine}, temperature: {self.temperature}, max_tokens: {self.max_tokens}, top_p: {self.top_p}, n: {self.n}, frequency_penelty: {self.frequency_penalty}, presence_penalty: {self.presence_penalty}, best_of: {self.best_of}, logprobs: {self.logprobs}"

    def get_multiple_sample(self, prompt_list: List[str]):
    
        while True:
            try:
                response = openai.Completion.create(
                    api_key = self.api_key,
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


def read_demons(path, example_id):
    demons = []
    fp = open(path, 'r')
    template = EXAMPLE_TEMPLATE[example_id]
    for p in fp.readlines():
        try:
            demon = json.loads(p.strip())
            dialogue, summary = demon['dialogue'], demon['summary'] 
            if example_id in [2, 3]:
                demons.append(template.format(s_a=s, s_b=t, score=score, exp=expl))
            else:
                demons.append(template.format(s_a=s, s_b=t, score=score))
        except Exception as e:
            logging.info(e)
            print("p")
        
    fp.close()
    return demons


def few_shot_evaluate_on_llm(input):
    prompt_path, request, testdataset, start, end, dump_path, task_description_id, example_template_id = input
    prompt_lists = [TASK_DESCRIPTION[task_description_id]]
    if os.path.exists(dump_path):
        print(f"File path {dump_path} exists")
        wf = open(dump_path, 'a')
    else:
        print(f"Create file {dump_path}")
        wf = open(dump_path, 'w+')

    demons = read_demons(prompt_path, example_template_id)
    print("demons length:", len(demons))
    prompt_lists.extend(demons)

    prompts = '\n'.join(prompt_lists)

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
        post_processing_response(resp, input_list, sample_id_list, golden_scores, wf)

    wf.close()

    golden_label_list, pred_label_list = read_dump_file(dump_path)
    score = metrics(golden_label_list, pred_label_list)

    wf = open(dump_path, 'a')
    wf.write(json.dumps(score)+"\n")

def read_api_keys(api_key_path):
    pattern = r"sk-.+$"
    api_keys = []
    raw_api_keys = open(api_key_path, 'r').readlines()
    for item in raw_api_keys:
        match = re.findall(pattern, item.strip())
        print(match)
        api_keys.append(match[0])
    
    return api_keys

def request_with_multi_keys(api_key_file, prompt_path, testdata_path, model_name, temperature, logprobs, dump_path, task_description_id, example_template_id, max_tokens):
    api_keys = read_api_keys(api_key_file)
    params = []
    test_datasets = read_datasets(testdata_path)
    api_keys_num = len(api_keys)
    split_data_num = math.ceil(len(test_datasets) / api_keys_num)
    for i, api_key in enumerate(api_keys):
        request = Request(
            api_key=api_key,
            engine=model_name,  
            temperature=temperature,  
            max_tokens=max_tokens,  
            top_p=1,  
            frequency_penalty=0,  
            presence_penalty=0,  
            logprobs=logprobs
        )
        
        tmp_start, tmp_end = i * split_data_num, min((i+1) * split_data_num, len(test_datasets))
        tmp_dump_path = f"{dump_path}{i}_task{task_description_id}_example{example_template_id}_{model_name}_temp{temperature}_maxtokens{max_tokens}.txt"
        if os.path.exists(tmp_dump_path):
            readed_lines = open(tmp_dump_path, 'r').readlines()
            for line in readed_lines:
                try:
                    if 'sample_id' in json.loads(line):
                        tmp_start = json.loads(line)['sample_id'] + 1
                except Exception as err:
                    logging.info(err)
                    print(line)
        
        params.append((prompt_path, request, test_datasets, tmp_start, tmp_end, tmp_dump_path, task_description_id, example_template_id))
    
    pool = Pool(api_keys_num)
    pool.map(few_shot_evaluate_on_llm, params)

    golden_list, pred_list = [], []
    for i in range(len(api_keys)):
        tmp_dump_path = f"{dump_path}{i}_task{task_description_id}_example{example_template_id}_{model_name}_temp{temperature}_maxtokens{max_tokens}.txt"
        fr = open(tmp_dump_path, 'r')

        for l in fr.readlines():
            try:
                d = json.loads(l)
                if "input" not in d.keys():
                    continue
                golden_list.append(d['golden'])
                pred_list.append(d['pred'])
            except:
                print(l)
    
    score = metrics(golden_list, pred_list)
    with open(tmp_dump_path, 'a') as fw:
        fw.write(json.dumps(score))


if __name__ == '__main__':
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser(description='Chinese Legal GPT')
    parser.add_argument('--testdata_path', type=str)
    parser.add_argument('--prompt_path', default='', type=str)
    parser.add_argument('--dump_path', default='', type=str)
    parser.add_argument('--k_shot', default=5, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=1000, type=int)
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--task_description_id', type=int, default=3)
    parser.add_argument('--example_template_id', type=int, default=1)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--logprobs', default=1)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--api_key_file', type=str, default='api_key.txt')
    parser.add_argument('--max_tokens', default=100, type=int)

    args = parser.parse_args()
    print(args)

    prompt_path = args.prompt_path
    testdata_path = args.testdata_path
    k_shot = args.k_shot
    dump_path = args.dump_path
    start = args.start
    end = args.end
    model_name = args.model_name
    task_description_id = args.task_description_id
    example_template_id = args.example_template_id
    temperature = args.temperature
    logprobs = args.logprobs
    api_key_file = args.api_key_file
    max_tokens = args.max_tokens
    
    request_with_multi_keys(args.api_key_file, prompt_path, testdata_path, model_name, temperature, logprobs, dump_path, task_description_id, example_template_id, max_tokens)