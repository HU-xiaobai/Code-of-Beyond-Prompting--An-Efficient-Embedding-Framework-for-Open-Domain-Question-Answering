# Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering

## Introduction

This repository accompanies our ACL 2025 long paper:

**Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering**  
Zhanghao Hu, Hanqi Yan, Qinglin Zhu, Zhenyi Shen, Yulan He, Lin Gui  
_In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025), Vienna, Austria_

📄 [Paper on ACL Anthology](https://aclanthology.org/2025.acl-long.981/) | 📄 [arXiv Preprint](https://arxiv.org/abs/2503.01606) 🔥[Project Page](https://zhanghao-acl25-embqa.github.io/ACL2025-EmbQA/)

> Large language models (LLMs) have recently pushed open-domain question answering (ODQA) to new frontiers. However, prevailing retriever–reader pipelines often depend on multiple rounds of prompt-level instructions, leading to high computational overhead, instability, and suboptimal retrieval coverage. In this paper, we propose **EmbQA**, an embedding-level framework that alleviates these shortcomings by enhancing both the retriever and the reader. Extensive experiments across three open-source LLMs, three retrieval methods, and four ODQA benchmarks demonstrate that EmbQA substantially outperforms recent baselines in both accuracy and efficiency.

---

## Installation

We recommend using the provided **sure_retrieval_env.yml** to recreate the exact environment used in our experiments:

```bash
# Recreate the exact environment we used
conda env create -f sure_retrieval_env.yml
conda activate sure_retrieval
```
This ensures full reproducibility of the results reported in our ACL 2025 paper.

💡 Note for users: Our environment file may include extra packages for our internal experiments.
If you prefer a minimal setup, you can create a fresh environment and install only the required packages manually:
```bash
conda create -n EmbQA python=3.11
conda activate EmbQA
```

## Datasets.zip (270 MB) can be downloaded from the GitHub Release Assets page. Please unzip it into ./data/ before running any scripts.


## Quick Start

Run inference with a single command. Here, we provide our method with LLaMA 3.1 8b Ins Model.

```bash
CUDA_VISIBLE_DEVICES=0 python query_openllm_ourframework_github.py --data_name hotpotqa (2wiki, nq, wq) --qa_data your_dataset_path --lm_type llama3 --n_retrieval 10 --infer_type embqa --output_folder your_output_dataset --end 500 (we follow Implementation of Harsh Trivedi et al. ACL 2023 and Kim et al. ICLR 2024 here) 
```

## Acknowledgements

This codebase is inspired by the [ICLR 2024 SuRe repository](https://github.com/bbuing9/ICLR24_SuRe).
.
We thank the authors for making their code publicly available, which helped us design and implement several components of EmbQA.

## Citation 
If you find this repository useful, please cite our paper:

```bash
@inproceedings{hu-etal-2025-beyond,
    title = "Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering",
    author = "Hu, Zhanghao  and
      Yan, Hanqi  and
      Zhu, Qinglin  and
      Shen, Zhenyi  and
      He, Yulan  and
      Gui, Lin",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.981/",
    doi = "10.18653/v1/2025.acl-long.981",
    pages = "19975--19990",
    ISBN = "979-8-89176-251-0"
}
```
