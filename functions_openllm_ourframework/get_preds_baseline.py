import json
import time
from functions_openllm_ourframework.gen_candidates import post_process_candidate, separation, use_api_candidate
from functions_openllm_ourframework.gen_summary import use_api_summary
from functions_openllm_ourframework.verification import use_api_verif, use_api_rank

import src_con.contriever
from src_con.dense_model import DenseEncoderModel

import numpy as np
import torch.nn as nn
import torch.optim as optim
import random

from data_utils import get_em_f1
import re

def build_prompt(input):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being factually coherent. If you don't know the answer to a question, please don't share false information."

    return f"{B_INST} {B_SYS}{SYSTEM_PROMPT}{E_SYS}{input} {E_INST} "

def get_query_candidate_self_attention(dataset, idx, n_articles=10, start_idx=0):
    data = dataset[idx]
    len_ctxs = len(data['contexts'])
    text = f"Below are {n_articles} passages related to the question at the end. After reading the passages, provide two correct candidates for the answer to the question at the end. Each answer should be in the form: (a) xx, (b) yy, and should not exceed 3 words for each candidate."

    for i in range(start_idx, start_idx + n_articles):
        idx_ctx = (i % len_ctxs)
        text += f"\n\nPassage #{i + 1} Title: {data['contexts'][idx_ctx]['title']}\nPassage #{i + 1} Text: {data['contexts'][idx_ctx]['text']}"

    text += f"\n\nQuestion:\n{data['question']}"

    text += f"\n\nEach answer should be in the form: (a) xx, (b) yy, and should not exceed 3 words for each candidate."
    text += f"\n\nAnswer: "
    text += f"<|reserved_special_token_9|>"
    return text


def extract_phrases(data):
    """Extract phrases following (a) and (b) from the candidate strings."""
    extracted_phrases = []
    for example in data:
        if isinstance(example, list) and example:
            text = example[0]

            # Extract (a) phrase - match everything up to (b)
            a_match = re.search(r"\(a\)\s(.*?)(?=\(b\))", text, re.DOTALL)
            a_phrase = a_match.group(1).strip().rstrip(',') if a_match else ""

            # Extract (b) phrase - match after (b)
            b_match = re.search(r"\(b\)\s(.*?)(?:[.,\n\[<]|<<|\]|</|$)", text, re.DOTALL)
            b_phrase = b_match.group(1).strip() if b_match else ""

            # Handle edge cases and ensure phrases are valid
            if a_phrase and b_phrase:
                extracted_phrases.append({"a": a_phrase, "b": b_phrase})
            else:
                extracted_phrases.append("no candidate")

    return extracted_phrases

def compute_entropy(logits):
    # logits shape [sequence_length, vocab_size]
    if torch.isnan(logits).any():
        print("Error: Logits contain NaN values on GPU!")
        raise ValueError("Logits contain NaN values!")

    if torch.isinf(logits).any():
        print("Error: Logits contain Inf values on GPU!")
        raise ValueError("Logits contain Inf values!")

    probs = torch.nn.functional.softmax(logits, dim=-1)  
    log_probs = torch.log(probs + 1e-12) 

    entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch_size, sequence_length]

    normalized_entropy = entropy.sum() / logits.size(0)  

    return normalized_entropy

import torch
import torch.nn.functional as F
import os

def memory_slot_representation_select(dataset, model, tokenizer, output_path, iteration):
    cross_attention_embeddings = []
    log_embedding_file_path = os.path.join(output_path, f'memory_slots_embedding_statistics_iteration_{iteration}.txt')
    for idx, example in enumerate(dataset):

        query_input = get_query_candidate_self_attention(dataset, idx, n_articles=10)
        if isinstance(query_input, list):
            prompt_input = [build_prompt(inp) for inp in query_input]
        else:
            prompt_input = [build_prompt(query_input)]

        try:
            input_ids = tokenizer(prompt_input, return_tensors="pt", add_special_tokens=True)["input_ids"]
        except RuntimeError as e:
            print(f"Skipping problematic input due to error: {e}")
            input_ids = torch.tensor([[tokenizer.pad_token_id]])

        try:
            input_ids = input_ids.to('cuda')
        except RuntimeError as e:
            print(f"Skipping problematic sample due to error: {e}")
            continue

        count = 0
        while True:
            count +=1
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True
                )

            all_hidden_states = outputs.hidden_states  # tuple, len = num_layers + 1
            context_hidden_state = all_hidden_states[-2]  # shape: (B, T, 4096)

            SPECIAL_TOKEN_ID = 128017  # llama3.1
            positions = (input_ids == SPECIAL_TOKEN_ID).nonzero(as_tuple=False)
            if positions.shape[0] == 0:
                print("No random token found in this prompt!")
                random_token_repr = None
            else:
                b_idx = positions[0, 0].item()
                t_idx = positions[0, 1].item()
                random_token_repr = context_hidden_state[b_idx, t_idx, :]
                variances = []

                for i in range(500):
                    token_vec = context_hidden_state[0, i, :]
                    sorted_vec_desc = np.sort(token_vec.cpu().to(torch.float32).numpy())[::-1]
                    top6 = sorted_vec_desc[1:7]

                    differences1 = top6[:-1] - top6[1:]  # shape: (5,)
                    std_val1 = np.std(differences1).item()
                    variances.append(std_val1)

                variances_sorted = sorted(variances, reverse=True)

                del context_hidden_state, all_hidden_states


            if random_token_repr is not None:

                sorted_values = random_token_repr.sort(descending=True).values
                top_six = sorted_values[1:7]  # shape: (6,)
                differences = top_six[:-1] - top_six[1:]  # shape: (5,)
                std_val = differences.std().item()

                # llama3.1
                if std_val < 0.05 or (count >50 and std_val <0.1):
                    cross_attention_embeddings.append(random_token_repr)
                    with open(log_embedding_file_path, "a", encoding="utf-8") as log_embedding_file:
                        log_embedding_file.write("--- Random Embedding Across Attention Scores Stds ---\n")
                        log_embedding_file.write(f"Example {idx}:\n")
                        log_embedding_file.write(f"Std = {std_val:.4f}\n")
                        log_embedding_file.write(f"Round count = {count}\n")
                    break
            else:
                print("hold on! you do not have random_token_repr ")


            del outputs

    return cross_attention_embeddings




def update_contain_answer_with_attention(dataset, sure_candidate, iteration, output_path, model, tokenizer):
    extracted_phrases = extract_phrases(sure_candidate)
    log_file_path = os.path.join(output_path, f'memory_slot_extracted_phrases_iter_{iteration}.txt')

    logits_path = os.path.join(output_path, f'memory_slot_logits_iter_{iteration}.pt')
    logits = torch.load(logits_path, map_location="cpu")
    prob_file_path = os.path.join(output_path, f'memory_slot_phrases_avg_prob_iter_{iteration}.txt')


    win_candidates = []

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        for idx, example in enumerate(dataset):
            if idx >= len(extracted_phrases):
                log_file.write(f"Warning: Mismatch at index {idx}\n")
                continue

            phrases = extracted_phrases[idx]
            if phrases == "no candidate":
                win_candidates.append(phrases)
                log_file.write(f"Example {idx}: no candidate\n")
                for passage in example["contexts"]:
                    passage["contain_answer"] = False

                continue


            a_phrase, b_phrase = phrases["a"], phrases["b"]

            sentence_logits = logits[idx].to('cuda')
            sentence_text = sure_candidate[idx][0]

            tokenized_sentence = tokenizer(sentence_text, return_offsets_mapping=True, return_tensors="pt")
            token_offsets = tokenized_sentence['offset_mapping'][0].cpu().numpy()

            def get_phrase_logits_old(phrase, text, logits, offsets):
                start_char = text.find(phrase)
                end_char = start_char + len(phrase)
                token_indices = [i for i, (s, e) in enumerate(offsets) if s >= start_char and e <= end_char]
                if not token_indices:
                    return None
                phrase_logits = logits[token_indices]  # [phrase_len, vocab_size]
                return phrase_logits


            a_logits = get_phrase_logits_old(a_phrase, sentence_text, sentence_logits, token_offsets)
            b_logits = get_phrase_logits_old(b_phrase, sentence_text, sentence_logits, token_offsets)

            if a_logits is not None:
                a_probs = F.softmax(a_logits, dim=-1)         # [phrase_len, vocab_size]
                a_top_probs, _ = a_probs.max(dim=-1)          # [phrase_len]
                a_avg_prob = a_top_probs.mean().item()
            else:
                a_avg_prob = 0.0

            if b_logits is not None:
                b_probs = F.softmax(b_logits, dim=-1)         # [phrase_len, vocab_size]
                b_top_probs, _ = b_probs.max(dim=-1)          # [phrase_len]
                b_avg_prob = b_top_probs.mean().item()
            else:
                b_avg_prob = 0.0

            with open(prob_file_path, "a", encoding="utf-8") as prob_file:
                prob_file.write(
                    f"Example {idx}: a_phrase={a_phrase}, b_phrase={b_phrase}, "
                    f"a_avg_prob={a_avg_prob:.4f}, b_avg_prob={b_avg_prob:.4f}\n"
                )

            def is_invalid_phrase(phrase):
                return len(phrase) > 100 or max([phrase.count(ch) for ch in set(phrase)]) > 50

            if is_invalid_phrase(a_phrase):
                winner = b_phrase
                log_file.write(
                    f"Example {idx}: winner={winner} a_phrase is invalid\n")
            elif is_invalid_phrase(b_phrase):
                winner = a_phrase
                log_file.write(
                    f"Example {idx}: winner={winner} b_phrase is invalid\n")
            elif a_logits is not None and b_logits is not None:
                a_entropy = compute_entropy(a_logits)
                b_entropy = compute_entropy(b_logits)
                winner = a_phrase if a_entropy < b_entropy else b_phrase

                log_file.write(
                    f"Example {idx}: winner={winner}, a_entropy={a_entropy:.4f}, b_entropy={b_entropy:.4f}\n")
            else:
                if a_logits is not None and b_logits is None:
                    winner = a_phrase
                    log_file.write(
                        f"Example {idx}: winner={winner}, a_logits is not None and b_logits is None \n")
                elif a_logits is None and b_logits is not None:
                    winner = b_phrase
                    log_file.write(
                        f"Example {idx}: winner={winner}, a_logits is None and b_logits is not None\n")
                else:
                    winner = a_phrase
                    log_file.write(
                        f"Example {idx}: winner={winner}, a_logits is None and b_logits is None\n")

            for passage in example["contexts"]:
                passage_text = passage["text"]
                if a_phrase in passage_text or b_phrase in passage_text:
                    passage["contain_answer"] = True
                else:
                    passage["contain_answer"] = False


            win_candidates.append(winner)

            del sentence_logits

    return dataset,win_candidates

def sure_infer_train(model, model_type, tokenizer, dataset ,start_time ,n_articles=10 ,output_path='./', num_iterations=5):
    print("=====> SuRe/Our framework Multi-Iteration Process")
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    all_token = 0

    torch.manual_seed(2025)

    win_candidates = None

    cross_attention_embeddings = None
    for iteration in range(num_iterations):
        print(f"=====> Iteration {iteration + 1}/{num_iterations}")
        print("=====> SuRe/Our framework Step #1. Random Embedding + Candidate Generation")

        path_to_candidate = output_path + f'/memory_slot_candidates_iter_{iteration}.json'
        logits_path = os.path.join(output_path, f'memory_slot_logits_iter_{iteration}.pt')
        all_token_path = os.path.join(output_path, f'token_consuming_{iteration}.json')

        if iteration != 0:
            dataset = json.load(open(updated_dataset_path))
            cross_attention_embeddings = memory_slot_representation_select(dataset ,model ,tokenizer ,output_path
                                                                           ,iteration)
            print(f"cross_attention_embeddings shape is {len(cross_attention_embeddings)}")
        if not os.path.exists(path_to_candidate):
            sure_candidate, logits ,all_token = use_api_candidate(model, model_type, tokenizer, dataset
                                                                  ,iteration=iteration
                                                                  ,cross_attention_embeddings=cross_attention_embeddings
                                                                  ,all_token=all_token ,n_articles=n_articles)

            with open(all_token_path, "a") as file:
                file.write(f"iteration {iteration} consuming token: {all_token}\n")

            with open(path_to_candidate, "w", encoding='utf-8') as writer:
                writer.write(json.dumps(sure_candidate, indent=4, ensure_ascii=False) + "\n")

            torch.save(logits, logits_path)
            print(f"logits saved at {logits_path}, shape: {logits.shape}")
            del logits
            torch.cuda.empty_cache()

        else:
            sure_candidate = json.load(open(path_to_candidate))

        result = analyze_candidates_contain_groundtruth(dataset, sure_candidate, output_path, iteration)

        metrics_file = os.path.join(output_path, f"metrics_iteration{iteration}.json")

        if os.path.exists(metrics_file):
            print(f"Metrics file already exists for iteration {iteration}: {metrics_file}")
            print("Skipping Step 1.5 this iteration...")
        else:
            print("=====> SuRe/Our framework Step #1.5. Extract Phrases and Update Retrieval Passages")
            dataset, win_candidates = update_contain_answer_with_attention(dataset, sure_candidate, iteration,
                                                                           output_path, model, tokenizer)
            print("=====> All Procedure is finished!")


            results_file = os.path.join(output_path, f"results_iteration{iteration}.json")
            with open(results_file, "w", encoding='utf-8') as writer:
                writer.write(json.dumps(win_candidates, indent=4, ensure_ascii=False) + "\n")

            em, f1 = get_em_f1(dataset, win_candidates)
            print("EM: {} F1: {}".format(em.mean(), f1.mean()))

            metrics_data = {
                "EM_mean": em.mean(),
                "F1_mean": f1.mean()
            }
            with open(metrics_file, "w", encoding='utf-8') as metrics_writer:
                json.dump(metrics_data, metrics_writer, indent=4, ensure_ascii=False)

            ans_idx = np.where(em == 1)[0]
            ans_idx_file = os.path.join(output_path, f"sure_ans_idx_{iteration}.npy")
            np.save(ans_idx_file, ans_idx)
            print(f"=====> Answer indices saved to {ans_idx_file}")

            end_time = time.time()
            end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))

            total_time_seconds = end_time - start_time
            total_time_minutes = total_time_seconds / 60
            total_time_hours = total_time_minutes / 60

            time_file_path = os.path.join(output_path, "time_document.txt")
            with open(time_file_path, "a") as file:
                file.write(f"iteration {iteration} ended/embedding level rerank framework start at: {end_time_str}\n")
                file.write(f"Total runtime: {total_time_seconds:.2f} seconds "
                           f"({total_time_minutes:.2f} minutes, {total_time_hours:.2f} hours)\n")

            if os.path.exists(logits_path):
                os.remove(logits_path)
                print(f"Deleted: {logits_path}")
            else:
                print(f"File not found: {logits_path}")


        sure_candidate = result["all_candidate_first_10_words"]

        query_mlp_path = os.path.join(output_path, f'memory_slot_query_mlp_iter_{iteration}.pt')
        query_last_mlp_path = os.path.join(output_path, f'memory_slot_query_mlp_iter_{iteration}.pt')
        candidate_mlp_path = os.path.join(output_path, f'memory_slot_candidate_mlp_iter_{iteration}.pt')
        candidate_last_mlp_path = os.path.join(output_path, f'memory_slot_candidate_mlp_iter_{iteration}.pt')
        updated_dataset_path = os.path.join(output_path, f'memory_slot_dataset_iter_{iteration + 1}.json')
        loss_log_path = os.path.join(output_path, f'memory_slot_loss_iter_{iteration}.txt')

        if os.path.exists(query_mlp_path) and os.path.exists(candidate_mlp_path):
            print(f"MLP parameters for iteration {iteration + 1} already exist. Skipping training...")

            if os.path.exists(updated_dataset_path):
                print(f"Re-ranked dataset for iteration {iteration + 1} already exists. Skipping Step 3...")
                continue
            else:
                print(f"Dataset not found. Proceeding to Step 3 for re-ranking...")
        else:
            print("=====> SuRe/Our framework Step #2. Retrieval Model and Training")


            retrieval_model, retrieval_tokenizer, _ = src_con.contriever.load_retriever('facebook/contriever-msmarco')
            retrieval_model = retrieval_model.cuda()
            retrieval_model.eval()

            query_encoder = retrieval_model
            doc_encoder = retrieval_model

            retrieval_model = DenseEncoderModel(query_encoder=query_encoder, doc_encoder=doc_encoder, tokenizer=retrieval_tokenizer)
            class MLP(nn.Module):
                def __init__(self, input_dim=768, hidden_dim=256, output_dim=768):
                    super(MLP, self).__init__()
                    self.fc1 = nn.Linear(input_dim, output_dim)


                def forward(self, x):
                    return self.fc1(x)

            query_mlp = MLP().cuda()
            candidate_mlp = MLP().cuda()

            if os.path.exists(query_last_mlp_path) and os.path.exists(candidate_last_mlp_path):
                query_mlp.load_state_dict(torch.load(query_last_mlp_path))
                candidate_mlp.load_state_dict(torch.load(candidate_last_mlp_path))

            optimizer = optim.Adam(list(query_mlp.parameters()) + list(candidate_mlp.parameters()), lr=1e-4)

            def multi_positive_infonce_loss(query_embeddings, positive_embeddings, negative_embeddings,
                                            temperature=0.07):
                pos_scores = torch.mm(query_embeddings, positive_embeddings.t()).squeeze(0) / temperature
                neg_scores = torch.mm(query_embeddings, negative_embeddings.t()).squeeze(0) / temperature
                logits = torch.cat([pos_scores, neg_scores], dim=0)
                labels = torch.cat([
                    torch.ones(len(pos_scores)),
                    torch.zeros(len(neg_scores))
                ]).cuda()

                loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), labels.unsqueeze(0))
                return loss

            with open(loss_log_path, "w") as loss_log_file:
                for epoch in range(15):
                    total_loss = 0
                    for idx, example in enumerate(dataset):
                        question = example["question"]
                        contexts = example["contexts"]
                        positive_contexts = [c for c in contexts if c["contain_answer"]]
                        negative_contexts = [c for c in contexts if not c["contain_answer"]]

                        if not positive_contexts:
                            continue

                        if not negative_contexts:
                            continue

                        num_negatives_needed = len(positive_contexts) * 5
                        if len(negative_contexts) > num_negatives_needed:
                            negative_contexts = random.sample(negative_contexts, num_negatives_needed)

                        query_embedding = retrieval_model.encode_queries([question], batch_size=1)
                        positive_embeddings = retrieval_model.encode_corpus(
                            [{"title": c["title"], "text": c["text"]} for c in positive_contexts],
                            batch_size=len(positive_contexts)
                        )
                        negative_embeddings = retrieval_model.encode_corpus(
                            [{"title": c["title"], "text": c["text"]} for c in negative_contexts],
                            batch_size=len(negative_contexts)
                        )


                        sure_candidate_dict = {example["question"]: sure_candidate[idx] for idx, example in
                                               enumerate(dataset)}
                        candidates = sure_candidate_dict.get(question, [])
                        candidate_embeddings = torch.tensor \
                            (retrieval_model.encode_queries(candidates, batch_size=len(candidates))).cuda()
                        query_embedding = torch.tensor(query_embedding).cuda()
                        query_embedding = query_mlp(torch.tensor(query_embedding).cuda())
                        candidate_embeddings = candidate_mlp(torch.tensor(candidate_embeddings).cuda())
                        query_embedding_new = query_embedding + candidate_embeddings.mean(dim=0)



                        positive_embeddings = torch.tensor(positive_embeddings).cuda()
                        negative_embeddings = torch.tensor(negative_embeddings).cuda()

                        loss = multi_positive_infonce_loss(query_embedding_new, positive_embeddings, negative_embeddings)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()

                    avg_loss = total_loss / len(dataset)
                    loss_log_file.write(f"Iteration {iteration + 1}, Epoch {epoch + 1}, Loss: {avg_loss:.4f}\n")
                    print(f"Iteration {iteration + 1}, Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

            torch.save(query_mlp.state_dict(), query_mlp_path)
            torch.save(candidate_mlp.state_dict(), candidate_mlp_path)


        print("=====> SuRe/Our framework Step #3. Re-ranking Retrieval Passages")

        def re_rank_passages(query_embedding_new, contexts):
            context_embeddings = retrieval_model.encode_corpus(
                [{"title": c["title"], "text": c["text"]} for c in contexts], batch_size=len(contexts)
            )
            if isinstance(context_embeddings, np.ndarray):
                context_embeddings = torch.tensor(context_embeddings)

            context_embeddings = context_embeddings.to(query_embedding_new.device)

            scores = torch.mm(query_embedding_new, context_embeddings.t()).squeeze(0).tolist()

            scored_contexts = list(zip(scores, contexts))

            scored_contexts.sort(key=lambda x: x[0], reverse=True)

            for idx, (score, context) in enumerate(scored_contexts):
                context["score"] = score

            return [sc[1] for sc in scored_contexts]

        for idx, example in enumerate(dataset):
            query = example["question"]
            contexts = example["contexts"]
            query_embedding = retrieval_model.encode_queries([query], batch_size=1)
            query_embedding = torch.tensor(query_embedding).cuda()
            query_embedding = query_mlp(torch.tensor(query_embedding).cuda())
            sure_candidate_dict = {example["question"]: sure_candidate[idx] for idx, example in
                                   enumerate(dataset)}
            candidates = [sure_candidate_dict.get(query, [])]
            candidate_embeddings = torch.tensor \
                (retrieval_model.encode_queries(candidates, batch_size=len(candidates))).cuda()
            candidate_embeddings = candidate_mlp(candidate_embeddings)
            query_embedding_new = query_embedding + candidate_embeddings.mean(dim=0)
            example["contexts"] = re_rank_passages(query_embedding_new, contexts)

        with open(updated_dataset_path, "w", encoding="utf-8") as writer:
            json.dump(dataset, writer, indent=4, ensure_ascii=False)

        end_time = time.time()
        end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))

        total_time_seconds = end_time - start_time
        total_time_minutes = total_time_seconds / 60
        total_time_hours = total_time_minutes / 60

        time_file_path = os.path.join(output_path, "time_document.txt")
        with open(time_file_path, "a") as file:
            file.write \
                (f"iteration {iteration} embedding-level rerank ended/Prompt embedding level start at: {end_time_str}\n")
            file.write(f"Total runtime: {total_time_seconds:.2f} seconds "
                       f"({total_time_minutes:.2f} minutes, {total_time_hours:.2f} hours)\n")

    return win_candidates

def sure_infer(model, model_type, tokenizer, dataset, n_articles=10, output_path='./'):
    print("=====> SuRe/Our framework Step #1. Random Embedding + Candidate Generation")

    path_to_candidate = output_path + '/{}'.format('candidates.json')
    if not os.path.exists(path_to_candidate):
        sure_candidate = use_api_candidate(model, model_type, tokenizer, dataset, n_articles=n_articles)
        with open(path_to_candidate, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(sure_candidate, indent=4, ensure_ascii=False) + "\n")
    else:
        sure_candidate = json.load(open(path_to_candidate))
    sure_candidate_post = post_process_candidate(sure_candidate)
    sure_candidate1, sure_candidate2 = separation(sure_candidate_post)

    print(f"sure_candidate_post is {sure_candidate_post}")
    print(f"sure_candidate1 is {sure_candidate1}")
    print(f"sure_candidate2 is {sure_candidate2}")



    print("=====> SuRe Step #2. Conditional Summarization")
    path_to_summary1 = output_path + '/{}'.format('summary1.json')
    if not os.path.exists(path_to_summary1):
        summary_candidate1 = use_api_summary(model, model_type, tokenizer, dataset, sure_candidate_post, pred=0, n_articles=n_articles)
        with open(path_to_summary1, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(summary_candidate1, indent=4, ensure_ascii=False) + "\n")
    else:
        summary_candidate1 = json.load(open(path_to_summary1))
    path_to_summary2 = output_path + '/{}'.format('summary2.json')
    if not os.path.exists(path_to_summary2):
        summary_candidate2 = use_api_summary(model, model_type, tokenizer, dataset, sure_candidate_post, pred=1, n_articles=n_articles)
        with open(path_to_summary2, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(summary_candidate2, indent=4, ensure_ascii=False) + "\n")
    else:
        summary_candidate2 = json.load(open(path_to_summary2))

    print("=====> SuRe Step #3. Self-Verification and Ranking")
    path_to_verfi1 = output_path + '/{}'.format('verif1.json')
    if not os.path.exists(path_to_verfi1):
        correctness_summary1 = use_api_verif(model, model_type, tokenizer, dataset, sure_candidate_post, summary_candidate1, pred_idx=0)
        with open(path_to_verfi1, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(correctness_summary1, indent=4, ensure_ascii=False) + "\n")
    else:
        correctness_summary1 = json.load(open(path_to_verfi1))

    path_to_verfi2 = output_path + '/{}'.format('verif2.json')
    if not os.path.exists(path_to_verfi2):
        correctness_summary2 = use_api_verif(model, model_type, tokenizer, dataset, sure_candidate_post, summary_candidate2, pred_idx=1)
        with open(path_to_verfi2, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(correctness_summary2, indent=4, ensure_ascii=False) + "\n")
    else:
        correctness_summary2 = json.load(open(path_to_verfi2))

    path_to_rank = output_path + '/{}'.format('ranking.npy')
    if not os.path.exists(path_to_rank):
        ranking_summary12 = use_api_rank(model, model_type, tokenizer, dataset, sure_candidate_post, summary_candidate1, summary_candidate2)
        np.save(path_to_rank, ranking_summary12)
    else:
        ranking_summary12 = np.load(path_to_rank)

    print(f"ranking_summary12 is {ranking_summary12}")
    sure_fin_preds, sure_fin_summary, all_indices = get_final_pred_sure(sure_candidate1, sure_candidate2, summary_candidate1, summary_candidate2, correctness_summary1, correctness_summary2, ranking_summary12)
    path_to_index = output_path + '/{}'.format('indices.json')
    with open(path_to_index, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(all_indices, indent=4, ensure_ascii=False) + "\n")
    path_to_summary = output_path + '/{}'.format('results_summary.json')
    with open(path_to_summary, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(sure_fin_summary, indent=4, ensure_ascii=False) + "\n")

    print(f"sure_fin_preds is {sure_fin_preds}")
    print(f"sure_fin_summary is {sure_fin_summary}")
    print(f"all_indices is {all_indices}")

    return sure_fin_preds

################## Functions to Get Prediction ##################
def analyze_candidates_contain_groundtruth(dataset, sure_candidate, output_path ,iteration):
    output_file = os.path.join(output_path, f"memory_slot_sure_analysis_output_{iteration}.txt")
    total_questions = len(dataset)
    fully_contain_count = 0
    partially_contain_count = 0
    all_candidate_first_10_words = []

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, data in enumerate(dataset):
            question = data.get("question", "")
            correct_answers = data.get("answers", [])

            if isinstance(correct_answers, str):
                correct_answers = [correct_answers.lower()]
            else:
                correct_answers = [answer.lower() for answer in correct_answers]

            if idx >= len(sure_candidate):
                candidate = ""
            else:
                current_candidates = sure_candidate[idx]
                if isinstance(current_candidates, list):
                    candidate = " ".join(current_candidates)
                else:
                    candidate = str(current_candidates)

            candidate_first_10_words = " ".join(candidate.split()[:10]).lower()
            all_candidate_first_10_words.append(candidate_first_10_words)

            is_fully_contain = all(answer in candidate_first_10_words for answer in correct_answers)
            if is_fully_contain:
                fully_contain_count += 1

            is_partially_contain = any(
                any(part in candidate_first_10_words for part in answer.split())
                for answer in correct_answers
            )
            if is_partially_contain:
                partially_contain_count += 1

            f.write(f"Question: {question}\n")
            f.write(f"Correct Answers: {correct_answers}\n")
            f.write(f"First 10 Words: {candidate_first_10_words}\n")
            f.write(f"Fully Contain: {is_fully_contain}\n")
            f.write(f"Partially Contain: {is_partially_contain}\n")
            f.write("\n")

    fully_contain_prob = fully_contain_count / total_questions if total_questions > 0 else 0
    partially_contain_prob = partially_contain_count / total_questions if total_questions > 0 else 0

    print(f"Probability of Fully Containing Correct Answers: {fully_contain_prob:.2f}")
    print(f"Probability of Partially Containing Correct Answers: {partially_contain_prob:.2f}")

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"Probability of Fully Containing Correct Answers: {fully_contain_prob:.2f}\n")
        f.write(f"Probability of Partially Containing Correct Answers: {partially_contain_prob:.2f}\n")

    return {
        "fully_contain_prob": fully_contain_prob,
        "partially_contain_prob": partially_contain_prob,
        "all_candidate_first_10_words": all_candidate_first_10_words
    }


def error_check_fin(res):
    corner_cases = ['Cannot', 'False', 'Unknown', 'N/A']

    for item in corner_cases:
        if item in res:
            return True
    return False

def get_final_pred_sure(candidate1, candidate2, summary1, summary2, correct1, correct2, ranking):
    n_sample = len(candidate1)
    n_choices = np.zeros(3)
    res, res_summary = [], []
    cand1_indices, cand2_indices, tie_indices = [], [], []
    for i in range(n_sample):
        rank_i = ranking[i]

        if ('True' in correct1[i] and 'True' in correct2[i]) or ('False' in correct1[i] and 'False' in correct2[i]):
            rank_i = rank_i
        elif 'True' in correct1[i] and error_check_fin(correct2[i]):
            rank_i = 0.5 * rank_i + 0.5 * np.array([1 ,0])
        elif error_check_fin(correct1[i]) and 'True' in correct2[i]:
            rank_i = 0.5 * rank_i + 0.5 * np.array([0 ,1])
        else:
            rank_i = rank_i

        max_vote = np.max(rank_i)

        # If Tie, then select the first candidate as answer
        if (rank_i == max_vote).sum() > 1:
            res.append([candidate1[i][0]])
            res_summary.append(summary1[i])
            n_choices[1] += 1
            tie_indices.append(i)
        else:
            select_idx = np.argmax(rank_i)
            if select_idx == 0:
                res.append([candidate1[i][0]])
                res_summary.append(summary1[i])
                n_choices[0] += 1
                cand1_indices.append(i)
            else:
                res.append([candidate2[i][0]])
                res_summary.append(summary2[i])
                n_choices[2] += 1
                cand2_indices.append(i)
    return res, res_summary, [cand1_indices, tie_indices, cand2_indices]

