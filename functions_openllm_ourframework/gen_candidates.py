import pprint
import json
import copy
import numpy as np
from tqdm import tqdm
import time
import openai

from datetime import timedelta, datetime
from functions_openllm_ourframework.common import api_query
import torch


def get_query_candidate(dataset, idx, n_articles=10, start_idx=0):
    data = dataset[idx]
    len_ctxs = len(data['contexts'])
    text = f"Below are {n_articles} passages related to the question at the end. After reading the passages, provide two correct candidates for the answer to the question at the end. Each answer should be in the form: (a) xx, (b) yy, and should not exceed 3 words for each candidate."

    for i in range(start_idx, start_idx + n_articles):
        idx_ctx = (i % len_ctxs)
        text += f"\n\nPassage #{i+1} Title: {data['contexts'][idx_ctx]['title']}\nPassage #{i+1} Text: {data['contexts'][idx_ctx]['text']}"

    text += f" <|reserved_special_token_9|> " #llama is 128017

    text += f"\n\nQuestion:\n{data['question']}"
    
    text += f"\n\nEach answer should be in the form: (a) xx, (b) yy, and should not exceed 3 words for each candidate."
    text += f"\n\nAnswer: "
    return text


def get_query_candidate_cross_attention(dataset, idx, n_articles=10, start_idx=0):
    data = dataset[idx]
    len_ctxs = len(data['contexts'])
    text = f"Below are {n_articles} passages related to the question at the end. After reading the passages, provide two correct candidates for the answer to the question at the end. Each answer should be in the form: (a) xx, (b) yy, and should not exceed 3 words for each candidate."

    for i in range(start_idx, start_idx + n_articles):
        idx_ctx = (i % len_ctxs)
        text += f"\n\nPassage #{i + 1} Title: {data['contexts'][idx_ctx]['title']}\nPassage #{i + 1} Text: {data['contexts'][idx_ctx]['text']}"

    text += f"\n\nQuestion:\n{data['question']}"

    text += f"\n\nEach answer should be in the form: (a) xx, (b) yy, and should not exceed 3 words for each candidate."
    text += f"\n\nAnswer: "
    text += f"<|reserved_special_token_8|>" #llama is 128016
    return text

def get_query_candidate_baseline(dataset, idx, n_articles=10, start_idx=0):
    data = dataset[idx]
    len_ctxs = len(data['contexts'])
    text = f"Below are {n_articles} passages related to the question at the end. After reading the passages, provide two correct candidates for the answer to the question at the end. Each answer should be in the form: (a) xx, (b) yy, and should not exceed 3 words for each candidate."

    for i in range(start_idx, start_idx + n_articles):
        idx_ctx = (i % len_ctxs)
        text += f"\n\nPassage #{i + 1} Title: {data['contexts'][idx_ctx]['title']}\nPassage #{i + 1} Text: {data['contexts'][idx_ctx]['text']}"


    text += f"\n\nQuestion:\n{data['question']}"

    text += f"\n\nEach answer should be in the form: (a) xx, (b) yy, and should not exceed 3 words for each candidate."
    text += f"\n\nAnswer: "
    return text

def use_api_candidate(model, model_type, tokenizer, dataset,iteration,cross_attention_embeddings, all_token,n_articles=10, start_idx=0):
    res = []
    res_logits = []
    queries = []

    if iteration == 0:
        for i, example in enumerate(dataset):
            query = get_query_candidate(dataset, i, n_articles, start_idx)
            queries.append(query)
    else:
        for i, example in enumerate(dataset):
            query = get_query_candidate_cross_attention(dataset, i, n_articles, start_idx)
            queries.append(query)

    for i, query in enumerate(tqdm(queries)):
        # The first time ground truth is just to take the position to use the random embedding to insert
        # and from the second time the ground truth position is to insert the "exploratory embedding" we called in the paper
        data = dataset[i]
        ground_truth = data['answers'][0]
        ground_truth = tokenizer(ground_truth, return_tensors="pt", add_special_tokens=False)['input_ids'].cuda()
        if iteration != 0:
            ground_truth = cross_attention_embeddings[i]
        with torch.no_grad():
            answer, logits,use_token = api_query(model, model_type, tokenizer, query, ground_truth=ground_truth)
            all_token+=use_token
        res.extend([[ans] for ans in answer])
        res_logits.append(logits.squeeze(0).cpu())
        del logits
        torch.cuda.empty_cache()

    from torch.nn.utils.rnn import pad_sequence
    res_logits = pad_sequence(res_logits, batch_first=True, padding_value=0)

    return res,res_logits,all_token


def use_api_candidate_baseline_random(model, model_type, tokenizer, dataset,cross_attention_embeddings, all_token,n_articles=10, start_idx=0):
    res = []
    queries = []

    for i, example in enumerate(dataset):
        query = get_query_candidate_cross_attention(dataset, i, n_articles, start_idx)
        queries.append(query)

    for query in tqdm(queries):
        ground_truth = cross_attention_embeddings[i]
        with torch.no_grad():
            answer,use_token = api_query(model, model_type, tokenizer, query, ground_truth=ground_truth)
            all_token+=use_token
        res.extend([[ans] for ans in answer])

    return res,all_token

def use_api_candidate_baseline(model, model_type, tokenizer, dataset, all_token, n_articles=10, start_idx=0):
    res = []
    queries = []

    for i, example in enumerate(dataset):
        query = get_query_candidate(dataset, i, n_articles, start_idx)
        queries.append(query)

    for query in tqdm(queries):
        data = dataset[i]
        ground_truth = data['answers'][0]
        ground_truth = tokenizer(ground_truth, return_tensors="pt", add_special_tokens=False)['input_ids'].cuda()
        answer,use_token= api_query(model, model_type, tokenizer, query, ground_truth=ground_truth)
        all_token+=use_token
        res.extend([[ans] for ans in answer])

    return res,all_token


def normalize_answer(s):
    chrs = [' ', ',', '.']
    while s[0] in chrs:
        s = s[1:]

    while s[-1] in chrs:
        s = s[:-1]
    
    return s

def divide_candidates(raw_candidates):
    res = []
    for item in raw_candidates:
        res_item = []
        raw_candidate = item[0]
        for i in range(4):
            try:
                target_symbol = chr(i + ord('a'))
                idx = raw_candidate.index(f'({target_symbol})')
                if i < 3: 
                    try:
                        next_symbol = chr(i + 1 + ord('a'))
                        idx_next = raw_candidate.index(f'({next_symbol})')
                        res_item.append(normalize_answer(raw_candidate[idx + len(target_symbol) + 2:idx_next]))
                    except:
                        res_item.append(normalize_answer(raw_candidate[idx + len(target_symbol) + 2:]))
                        break
                else:
                    res_item.append(normalize_answer(raw_candidate[idx + len(target_symbol) + 2:]))
            except:
                res_item.append(normalize_answer(raw_candidate))
                break
        res.append(res_item)
    return res

def handle_except(res_candidates, raw_candidates):
    for i, item in enumerate(res_candidates):
        if len(item) == 1:
            if len(item[0]) == 0:
                idx = raw_candidates[i][0].index(f'(a)')
                res_candidates[i] = raw_candidates[i][0][:idx]
            elif len(item[0].split(',')) > 4:
                print(i)
                print(item)
                res_candidates[i] = raw_candidates[i]
            else:
                new_res_candidate = []
                for split in res_candidates[i][0].split(','):
                    new_res_candidate.append(normalize_answer(split))
                res_candidates[i] = new_res_candidate
    return res_candidates

def get_choices_sampling(preds):
    choices = []
    avg_len = 0
    
    for pred in preds:
        choices_i = []

        for pred_i in pred:
            if pred_i not in choices_i:
                choices_i.append(pred_i)
        choices.append(choices_i)
        avg_len += len(choices_i)

    return choices

def post_process_candidate(raw_candidates):
    divided_candidates = divide_candidates(raw_candidates)
    res_candidates = handle_except(divided_candidates, raw_candidates)
    choices_candidates = get_choices_sampling(res_candidates)
    return choices_candidates

def separation(choices, n_choice=2):
    res = [] 
    for i in range(n_choice):
        res_i = []
        for choice in choices:
            if len(choice) > i:
                res_i.append([choice[i]])
            else:
                res_i.append(['N/A'])
        res.append(res_i)
    return res    

