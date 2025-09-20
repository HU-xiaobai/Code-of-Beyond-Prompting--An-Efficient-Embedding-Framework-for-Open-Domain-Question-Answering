import pprint
import json
import copy
import numpy as np
from tqdm import tqdm
import time
import openai
import torch

from datetime import timedelta, datetime

def api_query_batch(model, model_type, tokenizer, queries, batch_size=8, return_str=False):

    results = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'


    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]


        inputs = [(query) for query in batch_queries]
        tokenized = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )
        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()


        torch.cuda.empty_cache()

        generate_ids = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id
        )

        outputs = tokenizer.batch_decode(
            generate_ids.sequences[:, input_ids.shape[1]:],
            skip_special_tokens=True
        )


        outputs = ['N/A' if not ans.strip() else ans for ans in outputs]

        assert len(outputs) == len(batch_queries), f"Mismatch: outputs({len(outputs)}) vs batch_queries({len(batch_queries)})"

        results.append(outputs)

    return results


def use_api_base_batch(model, model_type, tokenizer, dataset, batch_size=8):

    queries = []

    print("=====> 数据处理并生成 Prompt...")
    for example in dataset:
        query = (
            f"{example['prompt']}\n"
            
            f"""Task description: predict the answer to the above question as short as you can. Remember: Do not explain, do not exceed 5 words and do not output "[INST]", "[/INST]","<<SYS>>", "<</SYS>>" tokens!"""
        )
        queries.append(query)

    print("=====> 开始批量推理...")
    results = api_query_batch(model, model_type, tokenizer, queries, batch_size=batch_size)

    return results


def build_prompt(input):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being factually coherent. If you don't know the answer to a question, please don't share false information."

    return f"{B_INST} {B_SYS}{SYSTEM_PROMPT}{E_SYS}{input} {E_INST} "


global_hidden_states = []

def api_query(model, model_type, tokenizer, query, return_str=False, ground_truth=None):
    if model_type == 'llama2' or 'llama3':
        if type(query) == list:
            inputs = [build_prompt(inp) for inp in query]
        else:
            inputs = [build_prompt(query)]
        input_ids = tokenizer(inputs, return_tensors="pt", add_special_tokens=True)['input_ids'].cuda()
    elif model_type == 'qwen2':
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being factually coherent. If you don't know the answer to a question, please don't share false information."},
            {"role": "user", "content": query}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_ids = tokenizer([text], return_tensors="pt").to(model.device)
    elif model_type == 'mistral':
        query = [{"role": "user", "content": query}]
        encodeds = tokenizer.apply_chat_template(query, return_tensors="pt")
        input_ids = encodeds.to("cuda")
    elif model_type == 'gemma':
        input_ids = tokenizer(query, return_tensors="pt", add_special_tokens=True)['input_ids'].cuda()
    else:
        raise ValueError


    generate_ids = model.generate(inputs=input_ids,
                                  return_dict_in_generate=True,
                                  do_sample=False,
                                  num_beams=1,
                                  max_new_tokens=512,
                                  output_scores=True,
                                  ground_truth = ground_truth,
                                  )

    logits = torch.stack(generate_ids.scores, dim=1)



    generated_sequences = generate_ids.sequences
    output_token_count = generated_sequences.shape[1] - input_ids.shape[1]
    del generated_sequences

    outputs = tokenizer.batch_decode(generate_ids.sequences[:, input_ids.shape[1]:], skip_special_tokens=True)

    for i, ans in enumerate(outputs):
        if ans.strip() == "":
            outputs[i] = 'N/A'

    del generate_ids
    torch.cuda.empty_cache()
    if return_str and len(outputs) == 1:
        return outputs[0],logits,output_token_count
    else:
        return outputs,logits,output_token_count


def get_query_baseline(dataset, idx, n_articles=10, start_idx=0):
    data = dataset[idx]
    len_ctxs = len(data['contexts'])
    if n_articles > 0:
        text = ""
        for i in range(start_idx, start_idx + n_articles):
            idx_ctx = (i % len_ctxs)
            text += f"Passage #{i + 1} Title: {data['contexts'][idx_ctx]['title']}\nPassage #{i + 1} Text: {data['contexts'][idx_ctx]['text']} \n\n"
        text += f"Task description: predict the answer to the following question. Do not exceed 3 words."
        text += f"\n\nQuestion: {data['question']}."
        text += f"\n\nAnswer: "
    else:
        text = f"Task description: predict the answer to the following question. Do not exceed 3 words."
        text += f"\n\nQuestion: {data['question']}."
        text += f"\n\nAnswer: "

    return text


def use_api_base(model, model_type, tokenizer, dataset, n_articles=10, start_idx=0):
    res = []
    queries = []

    for i, example in enumerate(dataset):
        query = get_query_baseline(dataset, i, n_articles, start_idx)
        queries.append(query)

    for query in tqdm(queries):
        answer,_ = api_query(model, model_type, tokenizer, query)
        res.extend([[ans] for ans in answer])

    return res