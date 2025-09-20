import os
import json
import argparse

import numpy as np
import sys

sys.path.insert(0, "/scratch/prj/inf_rate/xxxx/reason/Decoding") #change to your code
from transformers import AutoModelForCausalLM, AutoTokenizer

from functions_openllm_ourframework import use_api_base, sure_infer, sure_infer_train, use_api_base_batch
from data_utils import get_em_f1
import time
import torch

def preprocess_data_in_batches(dataset, batch_size=8):
    preprocessed_batches = []
    batch = []

    for i, sample in enumerate(dataset):
        context = "\n".join(
            [f"Title: {title}\n{sentence}" for title, sentence in
             zip(sample['context']['title'], sample['context']['sentences'])]
        )
        prompt = f"Context:\n{context}\n\nQuestion: {sample['question']}"

        batch.append({
            "id": sample['id'],
            "question": sample['question'],
            "context": context,
            "answers": sample['answer'],
            "prompt": prompt
        })

        if len(batch) == batch_size:
            preprocessed_batches.append(batch)
            batch = []

    if batch:
        preprocessed_batches.append(batch)

    return preprocessed_batches


import re


def parse_phrases(filepath):
    pattern = re.compile(r'^Example\s+(\d+):\s+a_phrase=(.*?),\s*b_phrase=(.*)$')
    parsed_results = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if "no candidate" in line:
                match_index = re.match(r'^Example\s+(\d+):', line)
                if match_index:
                    ex_idx = int(match_index.group(1))
                    parsed_results.append((ex_idx, "no candidate", "no candidate"))
                continue

            match = pattern.match(line)
            if match:
                ex_idx = int(match.group(1))
                a_phrase = match.group(2).strip()
                b_phrase = match.group(3).strip()
                parsed_results.append((ex_idx, a_phrase, b_phrase))

    return parsed_results


def preprocess_predictions(preds):
    processed_preds = []
    for pred in preds:
        if isinstance(pred, list):
            if pred:
                processed_preds.append(pred[0])
            else:
                processed_preds.append('')
        elif isinstance(pred, str):
            processed_preds.append(pred)
        else:
            raise ValueError(f"Unexpected prediction type: {type(pred)}, value: {pred}")
    return processed_preds


def get_accuracy(batches, preds):
    res_acc = []

    for item, pred in zip(batches, preds):
        ground_truths = item['answers']
        if not isinstance(ground_truths, list):
            ground_truths = [ground_truths]

        if isinstance(pred, list):
            pred = pred[0]

        match_found = any(gt in pred for gt in ground_truths)
        res_acc.append(1 if match_found else 0)

    return np.array(res_acc)


def truncate_predictions(preds, num_words=5):
    truncated_preds = []
    for pred in preds:
        if isinstance(pred, list):
            pred = pred[0]

        if isinstance(pred, str):
            words = pred.strip().split()
            truncated_pred = ' '.join(words[:num_words])
            truncated_preds.append(truncated_pred)
        else:
            truncated_preds.append(pred)

    return truncated_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query QA Data to GPT API.')
    parser.add_argument('--data_name', type=str, default='hotpot_qa', help='Name of QA Dataset')
    parser.add_argument('--qa_data', type=str, default=None, help='Path to QA Dataset (optional if using HF datasets)')
    parser.add_argument('--split', type=str, default='train', help='Dataset split (train, validation)')
    parser.add_argument('--start', type=int, default=None, help='Start index of QA Dataset')
    parser.add_argument('--end', type=int, default=None, help='End index of QA Dataset')
    parser.add_argument('--lm_type', type=str, default='llama3', help='Type of LLM (llama3, llama2, gemma, mistral)')
    parser.add_argument('--n_retrieval', type=int, default=10, help='Number of retrieval-augmented passages')
    parser.add_argument('--infer_type', type=str, default='sure', help='Inference Method (base or sure)',
                        choices=['base', 'sure'])
    parser.add_argument('--output_folder', type=str, default=None, help='Path for save output files')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for preprocessing and inference')
    parser.add_argument('--infer_only', type=bool, default=False, help='if we need train or just inference')

    args = parser.parse_args()

    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

    # Load QA Dataset
    print("=====> Data Load...")
    if args.qa_data:
        dataset = json.load(open(args.qa_data))
        start_idx, end_idx = args.start, args.end
        start_idx = 0 if start_idx is None else start_idx
        end_idx = len(dataset) if end_idx is None else end_idx
        if start_idx >= end_idx:
            raise ValueError("Start index must be less than end index.")
        dataset = dataset[start_idx:end_idx]
    else:
        raise ValueError("Either --data_name or --qa_data must be specified.")

    print("Number of QA Samples: {}".format(len(dataset)))


    if args.lm_type == "llama3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", padding_side='left')
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16,
                                                     attn_implementation="flash_attention_2")

    else:
        raise ValueError
    model = model.cuda()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.output_folder is None:
        raise ValueError("Output folder must be specified.")
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    method = f'{args.data_name}_{len(dataset)}example_{args.lm_type}_{args.infer_type}'
    method_folder = os.path.join(args.output_folder, method)
    os.makedirs(method_folder, exist_ok=True)

    time_file_path = os.path.join(method_folder, f'time_document.txt')
    with open(time_file_path, "a") as file:
        file.write(f"Program started at: {start_time_str}\n")

    print("=====> Begin Training (type: {})".format(args.infer_type))
    if not args.infer_only:
        if args.infer_type == 'base':
            all_results = use_api_base(model, args.lm_type, tokenizer, dataset, n_articles=args.n_retrieval)
        else:
            all_results = sure_infer_train(model, args.lm_type, tokenizer, dataset, start_time=start_time,
                                           n_articles=args.n_retrieval,
                                           output_path=method_folder)


    print("=====> All Procedure is finished!")
    results_file = os.path.join(method_folder, "results.json")  # 确保路径拼接正确
    with open(results_file, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(all_results, indent=4, ensure_ascii=False) + "\n")

    print(f"=====> Results saved to {results_file}")


    em, f1 = get_em_f1(dataset, all_results)
    print("EM: {} F1: {}".format(em.mean(), f1.mean()))

    if args.infer_type == 'base':
        accuracy = get_accuracy(dataset, all_results)
        print("Accuracy: {}".format(accuracy.mean()))
        metrics_file = os.path.join(method_folder, f"metrics_final.json")
        metrics_data = {
            "EM_mean": em.mean(),
            "F1_mean": f1.mean(),
            "Accuracy_mean": accuracy.mean()
        }
    else:
        metrics_file = os.path.join(method_folder, f"metrics_final.json")
        metrics_data = {
            "EM_mean": em.mean(),
            "F1_mean": f1.mean()
        }

    with open(metrics_file, "a", encoding='utf-8') as metrics_writer:
        json.dump(metrics_data, metrics_writer, indent=4, ensure_ascii=False)

    ans_idx = np.where(em == 1)[0]
    ans_idx_file = os.path.join(method_folder, f"{args.infer_type}_ans_idx.npy")
    np.save(ans_idx_file, ans_idx)
    print(f"=====> Answer indices saved to {ans_idx_file}")

    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))

    total_time_seconds = end_time - start_time

    total_time_minutes = total_time_seconds / 60
    total_time_hours = total_time_minutes / 60

    with open(time_file_path, "a") as file:
        file.write(f"Program ended at: {end_time_str}\n")
        file.write(f"Total runtime: {total_time_seconds:.2f} seconds "
                   f"({total_time_minutes:.2f} minutes, {total_time_hours:.2f} hours)\n")

