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
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import math


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from typing import List

import openai
import torch


TASK_DESCRIPTION = [
    """Given two sentences, the semantic textual similarity problem attempts to decide whether they are similar in meaning.
If the two sentences are the most similar, the text similarity between the two sentences is 5.0.
If the two sentences are the least similar, the text similarity between the two sentences is 0.0.
    """,
    """Given two sentences, the semantic textual similarity problem attempts to decide whether they are similar in meaning.
If the two sentences are the most similar, the text similarity between the two sentences is 5.0.
If the two sentences are the least similar, the text similarity between the two sentences is 1.0.
    """,
    """Given two sentences, the semantic textual similarity problem attempts to decide whether they are similar in meaning.
If the two sentences are the most similar, the text similarity between the two sentences is 5.0.
If the two sentences are the least similar, the text similarity between the two sentences is 1.0.
A text similarity score is a float number between 1.0 and 5.0.
    """,
    """Given two sentences, the semantic textual similarity problem attempts to decide whether they are similar in meaning.
If the two sentences are the most similar, the text similarity between the two sentences is 5.0.
If the two sentences are the least similar, the text similarity between the two sentences is 0.0.
A text similarity score is a float number between 0.0 and 5.0.
    """,
    """Task Instruction:
For this semantic textual similarity task, you will be given two pieces of text and asked to determine the degree of similarity between them. You will need to analyze the content of the text, including the words used, the context, and the overall meaning of the text, in order to determine the degree of similarity. You should rate the similarity on a scale of 0-5, with 0 being completely dissimilar and 5 being completely similar. A text similarity score is a float number between 1.0 and 5.0.
    """,#gpt3 generated,
    """Given two words, the lexical semantic similarity task attempts to decide whether they are similar in meaning.
If the two words are the most similar, the text similarity between the two sentences is 10.0.
If the two words are the least similar, the text similarity between the two sentences is 0.0.
A text similarity score is a float number between 0.0 and 10.0.
    """, #lexical semantic task
    ]

EXAMPLE_TEMPLATE = [
    "The text similarity between \"{s_a}\" and \"{s_b}\" is {score}.", 
    "Sentence A: {s_a}\nSentence B: {s_b}\nText similarity: {score}.",
    "Sentence A: {s_a}\nSentence B: {s_b}\nExplanation: {exp}\nText similarity: {score}.",
    "Sentence A: {s_a}\nSentence B: {s_b}\nText similarity: {score}.\nExplanation: {exp}",
    ]

GEN_REASON_TEMPLATE = [
    "Sentence A: {s_a}\nSentence B: {s_b}\nText similarity: {score}.\nPlease articulate clues and the reasoning process for determining the text similarity score between the two sentences.",
    "Sentence A: {s_a}\nSentence B: {s_b}\nText similarity: {score}.\nPlease articulate an explanation that leads to the text similarity score between the two sentences as {score}.",
    "Sentence A: {s_a}\nSentence B: {s_b}\nText similarity: {score}.\nThe explanation for the text similarity score between the two sentences as {score} is as follows:",
]

ANSWER_TEMPLATE = [
    "The text similarity between \"{s_a}\" and \"{s_b}\" is", 
    "Sentence A: {s_a}\nSentence B: {s_b}\nText similarity:",
    "Sentence A: {s_a}\nSentence B: {s_b}\nExplanation:",
    "Sentence A: {s_a}\nSentence B: {s_b}\nText similarity:",
    ]

NUMBER_PATTERN = r"\d+\.\d+"

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
            except openai.error.APIError as err:
                print("api error:")
                logging.exception(err)
                print(f"{self.api_key}")
                time.sleep(10)
            except openai.error.APIConnectionError as err:
                print("apiconnection error:")
                logging.exception(err)
                print(f"{self.api_key}")
                time.sleep(10)
            except openai.error.RateLimitError as err:
                print("ratelimit error:")
                logging.exception(err)
                print(f"{self.api_key}")
                time.sleep(10)
            except openai.error.InvalidRequestError as err:
                print("InvalidRequest error:")
                logging.exception(err)
                print(f"{self.api_key}")
                time.sleep(10)
            except openai.error.AuthenticationError as err:
                print("AuthenticationError error:")
                logging.exception(err)
                print(f"{self.api_key}")
                time.sleep(10)
            except openai.error.ServiceUnavailableError as err:
                print("ServiceUnavailableError error:")
                logging.exception(err)
                print(f"{self.api_key}")
                time.sleep(10)
            except Exception as err:
                print("other errors:")
                logging.exception(err)
                print(f"{self.api_key}")
                time.sleep(10)


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
    datasets = []
    with open(path, 'r') as fp:
        for line in fp.readlines():
            parts = line.strip().split('\t')
            s, t, score = parts[1], parts[2], float(parts[3])
            datasets.append((s, t, score))
    
    return datasets


def read_demons(path, example_id):
    demons = []
    fp = open(path, 'r')
    template = EXAMPLE_TEMPLATE[example_id]
    for p in fp.readlines():
        try:
            demon = json.loads(p.strip())
            s, t, score, expl = demon['Sentence A'], demon['Sentence B'], demon['Score'], demon['Explanations'].strip() 
            if example_id in [2, 3]:
                demons.append(template.format(s_a=s, s_b=t, score=score, exp=expl))
            else:
                demons.append(template.format(s_a=s, s_b=t, score=score))
        except Exception as e:
            logging.info(e)
            print("p")
        
    fp.close()
    return demons


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
                if example_template_id in [2]: # first reason, second score
                    num = float(match[-1])
                else:
                    num = float(match[0])
                sum += num
                count += 1

        if count == 0:
            print("Did not find number in response:", text)
            continue
            
        p_s = sum / count

        tmp = {"sample_id": sample_id_list[resp_id], "input": input_list[resp_id], "p_score": p_s,
               "g_score": golden_scores[resp_id], "model": resp['model'], "usage": resp["usage"], "choices": resp["choices"]}
        wf.write(json.dumps(tmp) + "\n")


def generate_few_shot_reasons_with_sampling(api_key, prompt_path, dump_path, task_description_id, example_template_id, gen_template_id):

    reason_request = Request(
        api_key=api_key,
        engine=model_name,  
        temperature=temperature,  
        max_tokens=150,  
        top_p=1,  
        n=cot_n,
        frequency_penalty=0,  
        presence_penalty=0,  
        best_of=cot_n,
        log_probs=logprobs
    )

    answer_request = Request(
        api_key=api_key,
        engine=model_name,  
        temperature=0,  
        max_tokens=150,  
        top_p=1,  
        n=cot_n,
        frequency_penalty=0,  
        presence_penalty=0,  
        best_of=cot_n,
        log_probs=logprobs
    )

    task = [TASK_DESCRIPTION[task_description_id]]
    if os.path.exists(dump_path):
        print(f"File path {dump_path} exists")
        wf = open(dump_path, 'a')
    else:
        print(f"Create file {dump_path}")
        wf = open(dump_path, 'w+')

    demons = read_demons(prompt_path, example_template_id)

    fp = open(prompt_path, 'r')
    template = GEN_REASON_TEMPLATE[gen_template_id]

    for p_idx, p in enumerate(fp.readlines()):
        prompt_list = task + demons[:p_idx] + demons[p_idx + 1:]
        prompts = '\n'.join(prompt_list)
        demon = json.loads(p.strip())
        s, t, score, expl = demon['Sentence A'], demon['Sentence B'], demon['Score'], demon['Explanations'].strip() 
        test_example = template.format(s_a=s, s_b=t, score=score)
        input = f"{prompts}\n{test_example}"
        print(f"test example: {test_example}")
        best_score = 1000
        for i in range(30):
            time.sleep(5)
            print("=" * 20 + f"{i} Try" + "=" * 20)
            reason_resp = reason_request.get_multiple_sample(prompt_list = [input])
            reason_text = reason_resp['choices'][0]['text'].strip()
            reason_text_wo_num = re.sub(NUMBER_PATTERN, "xx", reason_text)
            ##verify the generated reasons
            verify_example = EXAMPLE_TEMPLATE[2].format(s_a=s, s_b=t, exp=reason_text_wo_num, score="")
            verify_input = f"{task[0]}\n{verify_example}"
            verify_resp = answer_request.get_multiple_sample(prompt_list = [verify_input])
            verify_answer_text = verify_resp['choices'][0]['text'].strip()
            print("verified answer:", verify_answer_text)
            if abs(float(verify_answer_text) - score) < 0.3:
                prompt_dict = {"Sentence A": s, "Sentence B": t, "Score": score, "Explanations": reason_text, "ReasonInput": input, "Model": model_name, "VerifyInput": verify_input, "ReasonOutput": reason_text, "VerifyOutput": verify_answer_text}
                break
            if abs(float(verify_answer_text) - score) < best_score:
                prompt_dict = {"Sentence A": s, "Sentence B": t, "Score": score, "Explanations": reason_text, "ReasonInput": input, "Model": model_name, "VerifyInput": verify_input, "ReasonOutput": reason_text, "VerifyOutput": verify_answer_text}
                best_score = abs(float(verify_answer_text) - score)

        wf.write(json.dumps(prompt_dict) + "\n")
    
    wf.close()

def generate_few_shot_reasons(api_key, prompt_path, dump_path, task_description_id, example_template_id, gen_template_id):

    reason_request = Request(
        api_key=api_key,
        engine=model_name,  
        temperature=0,  
        max_tokens=150,  
        top_p=1,  
        n=cot_n,
        frequency_penalty=0,  
        presence_penalty=0,  
        best_of=cot_n
    )

    task = [TASK_DESCRIPTION[task_description_id]]
    if os.path.exists(dump_path):
        print(f"File path {dump_path} exists")
        wf = open(dump_path, 'a')
    else:
        print(f"Create file {dump_path}")
        wf = open(dump_path, 'w+')

    demons = read_demons(prompt_path, example_template_id)

    fp = open(prompt_path, 'r')
    template = GEN_REASON_TEMPLATE[gen_template_id]

    for p_idx, p in enumerate(fp.readlines()):
        prompt_list = task + demons[:p_idx] + demons[p_idx + 1:]
        prompts = '\n'.join(prompt_list)
        demon = json.loads(p.strip())
        s, t, score, expl = demon['Sentence A'], demon['Sentence B'], demon['Score'], demon['Explanations'].strip() 
        test_example = template.format(s_a=s, s_b=t, score=score)
        input = f"{prompts}\n{test_example}"
        print("=" * 20 + f"{p_idx} prompt" + "=" * 20)
        reason_resp = reason_request.get_multiple_sample(prompt_list = [input])
        reason_text = reason_resp['choices'][0]['text'].strip()
        prompt_dict = {"Sentence A": s, "Sentence B": t, "Score": score, "Explanations": reason_text, "ReasonInput": input, "Model": model_name, "finish_reason": reason_resp["choices"][0]["finish_reason"]}
        wf.write(json.dumps(prompt_dict) + "\n")
        time.sleep(5)
    
    wf.close()

def few_shot_evaluate_on_gpt3(input):
    prompt_path, request, testdataset, start, end, dump_path, task_description_id, example_template_id, cot_n = input
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

    wf = open(dump_path, 'a')
    wf.write(json.dumps({"spearman_corr": spearman_corr})+"\n")

def read_api_keys(api_key_path):
    pattern = r"sk-.+$"
    api_keys = []
    raw_api_keys = open(api_key_path, 'r').readlines()
    for item in raw_api_keys:
        match = re.findall(pattern, item.strip())
        print(match)
        api_keys.append(match[0])
    
    return api_keys

def request_with_multi_keys(api_key_file, prompt_path, testdata_path, model_name, temperature, cot_n, logprobs, dump_path, task_description_id, example_template_id, max_tokens):
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
            n=cot_n,
            frequency_penalty=0,  
            presence_penalty=0,  
            best_of=cot_n,
            logprobs=logprobs
        )
        print(request)
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
        
        params.append((prompt_path, request, test_datasets, tmp_start, tmp_end, tmp_dump_path, task_description_id, example_template_id, cot_n))
    
    pool = Pool(api_keys_num)
    pool.map(few_shot_evaluate_on_gpt3, params)

    g_s, p_s = [], []
    for i in range(len(api_keys)):
        tmp_dump_path = f"{dump_path}{i}_task{task_description_id}_example{example_template_id}_{model_name}_temp{temperature}_maxtokens{max_tokens}.txt"
        fr = open(tmp_dump_path, 'r')

        for l in fr.readlines():
            try:
                d = json.loads(l)
                if "input" not in d.keys():
                    continue
                g_s.append(d['g_score'])
                p_s.append(d['p_score'])
            except:
                print(l)
    
    print("test data size:", len(g_s))
    with open(tmp_dump_path, 'a') as fw:
        print(f"The final spearman correlation is {spearmanr(g_s, p_s)}, pearson correlation is {pearsonr(g_s, p_s)}")
        fw.write(json.dumps({"final_spearman_corr": spearmanr(g_s, p_s), "final_pearson_corr": pearsonr(g_s, p_s)}))


if __name__ == '__main__':
    print(datetime.datetime.now())
    BASE = "/data/ganleilei/law/ContrastiveLJP"
    parser = argparse.ArgumentParser(description='GPT3 for text similarity')
    parser.add_argument('--testdata_path', type=str)
    parser.add_argument('--prompt_path', default='', type=str)
    parser.add_argument('--dump_path', default='', type=str)
    parser.add_argument('--k_shot', default=25, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=1000, type=int)
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--task_description_id', type=int, default=3)
    parser.add_argument('--example_template_id', type=int, default=1)
    parser.add_argument('--reason_gen_template_id', type=int, default=2)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--cot_n', default=1, type=int)
    parser.add_argument('--logprobs', default=1)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--api_key_file', type=str, default='api_key.txt')
    parser.add_argument('--max_tokens', default=15, type=int)


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
    reason_gen_template_id = args.reason_gen_template_id
    temperature = args.temperature
    cot_n = args.cot_n
    logprobs = args.logprobs
    api_key_file = args.api_key_file
    max_tokens = args.max_tokens
    multi, single_key, gen_explanations = True, False, False

    if multi:
        request_with_multi_keys(args.api_key_file, prompt_path, testdata_path, model_name, temperature, cot_n, logprobs, dump_path, task_description_id, example_template_id, max_tokens)

    elif single_key:
        api_key = args.api_key
        test_datasets = read_datasets(testdata_path)
        request = Request(
                api_key=api_key,
                engine=model_name,  
                temperature=temperature,  
                max_tokens=150,  
                top_p=1,  
                n=cot_n,
                frequency_penalty=0,  
                presence_penalty=0,  
                best_of=cot_n,
                log_probs=logprobs
            )
        input = (prompt_path, request, test_datasets, start, end, k_shot, dump_path, task_description_id, example_template_id, cot_n)
        few_shot_evaluate_on_gpt3(input)

    elif gen_explanations:
        api_key = args.api_key
        generate_few_shot_reasons(api_key, prompt_path, dump_path, task_description_id, example_template_id, reason_gen_template_id)
        # generate_few_shot_reasons_with_sampling(api_key, prompt_path, dump_path, task_description_id, example_template_id, reason_gen_template_id)