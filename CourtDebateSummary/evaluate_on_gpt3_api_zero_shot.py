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
    """Given two sentences, the semantic textual similarity problem attempts to decide whether they are similar in meaning.
If the two sentences are the most similar, the text similarity between the two sentences is 5.0.
If the two sentences are the least similar, the text similarity between the two sentences is 0.0."""
    ]

ANSWER_TEMPLATE = [
    "The text similarity between \"{s_a}\" and \"{s_b}\" is", 
    "Sentence A: {s_a}\nSentence B: {s_b}\nText similarity:",
    "Sentence A: {s_a}\nSentence B: {s_b}\nPlease first articulate clues and the reasoning process for determining the similarity score between the two sentences. Next, based on the clues and the reasoning process, assign the similarity score between the two sentences.",
    "Sentence A: {s_a}\nSentence B: {s_b}\nPlease first articulate clues (positive or negative) and the reasoning process for determining the similarity score between the two sentences. Next, based on the clues and the reasoning process, assign the similarity score between the two sentences.",
    "Sentence A: {s_a}\nSentence B: {s_b}\nPlease first assign the similarity score between the two sentences. Next, articulate clues and the reasoning process for determining the similarity score.",
    "Word A: {s_a}\nWord B: {s_b}\nSimilarity Score:",
    "Word A: {s_a}\nWord B: {s_b}\nPlease first list the synonyms for the two words. Next, based on the two words and the synonyms, give the similarity score between the two words.",
    ]

NUMBER_PATTERN = r"\d+\.\d+"

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
                print("request error:")
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
    g_s, p_s = [], []

    for l in fr.readlines():
        try:
            d = json.loads(l)
        except:
            print("cannot parse line:", l)

        if "input" not in d.keys():
            continue
        g_s.append(d['g_score'])
        p_s.append(d['p_score'])
    fr.close()

    return g_s, p_s


def post_processing_response(resp, input_list, sample_id_list, golden_scores, wf, cot_n, example_template_id):
    for resp_id in range(len(input_list)):
        text = [item['text'].strip() for item in resp['choices'][resp_id*cot_n: resp_id*cot_n + cot_n]]
        sum, count = 0, 0
        for single_res in text:
            match = re.findall(NUMBER_PATTERN, single_res)
            if len(match) >= 1:
                if example_template_id in [2,3]: # first reason, second score
                    num = float(match[-1])
                else:
                    num = float(match[0])
                sum += num
                count += 1

        if count == 0:
            print("Did not find number in response:", text)
            continue
            
        p_s = sum / count

        tmp = {"sample_id": sample_id_list[resp_id], "input": input_list[resp_id],
                    "response": resp['choices'][resp_id*cot_n: resp_id*cot_n + cot_n], "p_score": p_s, "g_score": golden_scores[resp_id], "model": resp['model']}
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
    prompt_lists = [TASK_DESCRIPTION[task_description_id]]
    if os.path.exists(dump_path):
        print(f"File path {dump_path} exists")
        wf = open(dump_path, 'a')
    else:
        print(f"Create file {dump_path}")
        wf = open(dump_path, 'w+')
    prompts = '\n'.join(prompt_lists)

    input_list, sample_id_list, golden_scores = [], [], []
    for sample_id in range(start, end):
        s, t, g_s = testdataset[sample_id]
        template = ANSWER_TEMPLATE[example_template_id]
        test_example = template.format(s_a=s, s_b=t)
        input = f"{prompts}\n{test_example}"
        input_list.append(input)
        sample_id_list.append(sample_id)
        golden_scores.append(g_s)
        if len(input_list) == 5:
            resp = request.get_multiple_sample(input_list)
            post_processing_response(resp, input_list, sample_id_list, golden_scores, wf, cot_n, example_template_id)
            time.sleep(5)
            print("#", end="", flush=True)
            input_list, sample_id_list, golden_scores = [], [], []

    if len(input_list):
        resp = request.get_multiple_sample(input_list)
        post_processing_response(resp, input_list, sample_id_list, golden_scores, wf, cot_n, example_template_id)

    wf.close()

    g_scores, pred_scores = read_dump_file(dump_path)
    spearman_corr = spearmanr(g_scores, pred_scores)
    print("spearman correlation:", spearman_corr)

    wf = open(dump_path, 'a')
    wf.write(json.dumps({"spearman_corr": spearman_corr})+"\n")


if __name__ == '__main__':
    print(datetime.datetime.now())
    BASE = "/data/ganleilei/law/ContrastiveLJP"
    parser = argparse.ArgumentParser(description='GPT3 for text similarity')
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