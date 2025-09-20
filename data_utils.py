import csv
import json
import logging
import os
import regex
import argparse
import re
import string
import sys
import unicodedata
import datasets
import numpy as np

from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Sequence, Tuple, Union

logger = logging.getLogger("data_utils")

class SimpleTokenizer:

    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self, **kwargs):
        self._regexp = regex.compile(
            f"({self.ALPHA_NUM})|({self.NON_WS})", flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def _amend(self, text: str) -> str:
        text = text.replace("\u2019", "'").replace("--", "_")

        for special_word in ("don't", "ain't", "won't", "didn't", "can't", "doesn't", "wanna", "gonna", "gimme"):
            if special_word in text:
                if special_word.endswith("nna"):
                    fixed = f"{special_word[:-len('na')]} na"
                elif special_word.endswith("mme"):
                    fixed = f"{special_word[:-len('me')]} me"
                else:
                    fixed = special_word[: -len("n't")] + " n't"

                text = text.replace(special_word, fixed)

        return text

    def tokenize(self, text: str, as_string: bool = False) -> Union[str, List[str]]:
        text = self._amend(text)

        tokens = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            tokens.append(matches[i].group())

        if as_string:
            return " ".join(tokens)
        else:
            return tokens


@dataclass
class Question:
    text: str
    answers: Union[Set[str], List[str]]
    id: Optional[str] = None
    tokens: Optional[List[str]] = field(default=None)
    acceptable_answers: Optional[List[str]] = field(default=None)
    unacceptable_answers: Optional[List[str]] = field(default=None)

    @property
    def has_answers(self) -> bool:
        return self.answers and len(self.answers) > 0
    
    @property
    def has_annotated_answers(self) -> bool:
        return len(self.gold_answers) > 0 or self.unacceptable_answers

    @property
    def tokenized_text(self) -> Optional[str]:
        return " ".join(self.tokens) if self.tokens is not None else None

    def update_answers(self, annotated_answers):
        if not annotated_answers:
            return

        self.acceptable_answers = annotated_answers["yes"]
        self.unacceptable_answers = annotated_answers["no"]

    def is_unacceptable(self, candidate_answer: str) -> bool:
        if self.unacceptable_answers:
            for ans in self.unacceptable_answers:
                print(f"ans is {ans}")
                print(f"candidate_answer is {candidate_answer}")
                if candidate_answer == ans or candidate_answer.lower() == ans.lower():
                    return True

        return False

    @property
    def gold_answers(self) -> Set[str]:
        answers = set(self.answers) if self.answers else set()
        
        if self.acceptable_answers:
            answers.update(self.acceptable_answers)

        if self.unacceptable_answers:
            for a in self.unacceptable_answers:
                if a in answers:
                    answers.remove(a)
                elif a.lower() in answers:
                    answers.remove(a.lower())
                    
        return answers

    def to_json(self) -> Dict[str, Any]:
        json_dict = dict(
            question=self.text,
            id=self.id,
            answers=self.answers,
        )

        return json_dict

    @classmethod
    def from_json(cls, q_dict, idx: int = 0):
        return Question(
            q_dict["question"],
            q_dict.get("answer", q_dict.get("answers", None)),
            q_dict.get("id", idx),
        )


def read_json(dataset_path: os.PathLike) -> Iterable[Question]:
    tokenizer = SimpleTokenizer()

    with open(dataset_path, "r", encoding="utf-8") as reader:
        dataset = json.load(reader)
    for i, q in enumerate(dataset):
        tokens = tokenizer.tokenize(q["question"])
        yield Question(q["question"], q["answers"], str(i), tokens=tokens)


def reader_tsv(dataset_path: os.PathLike) -> Iterable[Question]:
    tokenizer = SimpleTokenizer()

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(dataset_reader):
            tokens = tokenizer.tokenize(row[0].strip())

            try:
                answers = [a.strip() for a in eval(row[1])]
            except:
                answers = row[1].strip().split(" | ")

            yield Question(row[0].strip(), answers, str(i), tokens=tokens)


def read_jsonl(dataset_path: os.PathLike) -> Iterable[Question]:
    tokenizer = SimpleTokenizer()

    with open(dataset_path, "r", encoding="utf-8") as dataset_reader:
        for i, line in enumerate(dataset_reader):
            if not line.strip():
                continue

            q_dict = json.loads(line)
            yield Question(
                q_dict["question"],
                q_dict.get("answer", q_dict.get("answers", None)),
                q_dict.get("id", str(i)),
                tokenizer.tokenize(q_dict["question"]),
            )


def load_dataset(dataset_name: str, split: str = "validation"):
    tokenizer = SimpleTokenizer()
    for i, ex in enumerate(datasets.load_dataset(dataset_name, split=split)):
        answers = ex["answer"]
        if isinstance(answers, dict):
            answers = [answers["value"]] + [answers["aliases"]]
        yield Question(
            ex["question"],
            answers,
            ex.get("question_id", str(i)),
            tokenizer.tokenize(ex["question"]),
        )


def read_questions(dataset_file: os.PathLike, split: str = "validation") -> Iterable[Question]:
    dataset_path = Path(dataset_file)
    if dataset_path.exists():
        if dataset_path.suffix == ".json":
            return list(read_json(dataset_path))
        elif dataset_path.suffix in (".tsv", ".txt"):
            return list(reader_tsv(dataset_path))
        elif dataset_path.suffix == ".jsonl":
            return list(read_jsonl(dataset_path))
        else:
            raise ValueError(f"Unknown file format: {dataset_path.suffix}")
    else:
        return list(datasets.load_dataset(dataset_file, split))


def read_predict_file(
    predict_file: os.PathLike,
    questions: Optional[Sequence[Question]] = None,
) -> Tuple[Mapping[str, str], Sequence[Question]]:
    tokenizer = SimpleTokenizer()
    predict_file = Path(predict_file)

    collected_questions = []
    predicted_dict = {}
    if predict_file.suffix in (".tsv", ".csv", ".txt"):
        assert questions, "dataset file must be provided"

        with predict_file.open("r") as p:
            reader = csv.reader(p, delimiter="," if predict_file.suffix == ".csv" else "\t")

            for predicted_row, q in zip(reader, questions):
                predicted_dict[q.tokenized_text.lower()] = predicted_row[1].strip()
    else:
        if predict_file.suffix == ".json":
            with predict_file.open("r") as p:
                predictions = json.load(p)
        else:
            with predict_file.open("r") as p:
                predictions = [json.loads(line) for line in p if line.strip()]

        answer_dict = {}
        for predicted_item in predictions:
            predicted_method = predicted_item["prediction"]
            if isinstance(predicted_method, (list, tuple)):
                predicted_answer = predicted_method[0].strip()
            else:
                predicted_answer = predicted_method.strip()

            q = Question(
                predicted_item["question"],
                predicted_item["answer"],
                tokens=tokenizer.tokenize(predicted_item["question"]),
            )
            if not questions:
                collected_questions.append(q)
            answer_dict[q.tokenized_text.lower()] = predicted_answer

        if questions:
            for q in questions:
                if q.tokenized_text.lower() in answer_dict:
                    predicted_dict[q.tokenized_text.lower()] = answer_dict[q.tokenized_text.lower()]
        else:
            questions = collected_questions
            predicted_dict = answer_dict

    return predicted_dict, questions


def read_annotations(annotation_file: os.PathLike) -> Mapping[str, Mapping[str, Set[str]]]:
    tokenizer = SimpleTokenizer()
    annotated_answers = defaultdict(lambda: {"yes": set(), "no": set()})

    with open(annotation_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            question = tokenizer.tokenize(row["Question"], as_string=True).lower()
            if "Acceptable?" not in row or not row["Acceptable?"].strip():
                continue

            annotated_answers[question][row["Acceptable?"].strip().lower()].add(row["Model answer"])
            if question.endswith("?"):
                annotated_answers[question[:-1].strip()][row["Acceptable?"].strip().lower()].add(row["Model answer"])

    logger.info(f"Annotations loaded with {len(annotated_answers)} entries")
    return dict(annotated_answers)


def _normalize(text):
    return unicodedata.normalize('NFD', text)

def normalize_answer(s):
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def regex_match(text, pattern):
    try:
        pattern = re.compile(
            _normalize(pattern),
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(_normalize(text)) is not None


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truth):
    score = metric_fn(prediction, ground_truth)
    return score

def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = "Unanswered question " + qa["id"] + " will receive score 0."
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]
                exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}

def em_eval(ground_truth, candidate_answer: str, match: str = "string") -> int:
    return int(
        metric_max_over_ground_truths(
            regex_match if match == "regex" else exact_match_score,
            candidate_answer,
            ground_truth,
        )
    )

def f1_eval(ground_truth, candidate_answer: str) -> float:

    return metric_max_over_ground_truths(
        f1_score,
        candidate_answer,
        ground_truth,
    )


import numpy as np


def get_em_f1(batches, preds):
    res_em = []
    res_f1 = []

    for item, pred in zip(batches, preds):
        ground_truths = item['answers']
        if not isinstance(ground_truths, list):
            ground_truths = [ground_truths]

        if isinstance(pred, list):
            pred = pred[0]

        best_em = -1
        best_f1 = -1

        for gt in ground_truths:
            curr_em = em_eval(gt, pred)
            curr_f1 = f1_eval(gt, pred)

            if curr_em > best_em:
                best_em = curr_em
                best_f1 = curr_f1
            elif curr_em == best_em and curr_f1 > best_f1:
                best_f1 = curr_f1

        res_em.append(best_em)
        res_f1.append(best_f1)

    return np.array(res_em), np.array(res_f1)


def get_em_f1_old(batches, preds):
    res_em = []
    res_f1 = []


    for item_idx, (item, pred) in enumerate(zip(batches, preds)):

        ground_truth = item['answers']
        if isinstance(pred, list):
            pred = pred[0]
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0]

        em = em_eval(ground_truth, pred)
        f1 = f1_eval(ground_truth, pred)
        res_em.append(em)
        res_f1.append(f1)


    return np.array(res_em), np.array(res_f1)



def get_em_f1_oracle(dataset, choices):
    res_em = []
    res_f1 = []
    for i, item in enumerate(dataset):
        q = Question(item['question'], item['answers'])
        em, f1 = 0, 0
        for _, choice in enumerate(choices[i]):
            em_j = em_eval(q, choice)
            f1_j = f1_eval(q, choice)
            if em_j > em:
                em = em_j
            if f1_j > f1:
                f1 = f1_j
        res_em.append(em)
        res_f1.append(f1)
    return np.array(res_em), np.array(res_f1)