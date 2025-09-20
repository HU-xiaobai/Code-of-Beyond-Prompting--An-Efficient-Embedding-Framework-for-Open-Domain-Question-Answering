# Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering

## Introduction

This repository accompanies our ACL 2025 long paper:

**Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering**  
Zhanghao Hu, Hanqi Yan, Qinglin Zhu, Zhenyi Shen, Yulan He, Lin Gui  
_In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025), Vienna, Austria_

📄 [Paper on ACL Anthology](https://aclanthology.org/2025.acl-long.981/) | 📄 [arXiv Preprint](https://arxiv.org/abs/2505.24688)

> Large language models (LLMs) have recently pushed open-domain question answering (ODQA) to new frontiers. However, prevailing retriever–reader pipelines often depend on multiple rounds of prompt-level instructions, leading to high computational overhead, instability, and suboptimal retrieval coverage. In this paper, we propose **EmbQA**, an embedding-level framework that alleviates these shortcomings by enhancing both the retriever and the reader. Extensive experiments across three open-source LLMs, three retrieval methods, and four ODQA benchmarks demonstrate that EmbQA substantially outperforms recent baselines in both accuracy and efficiency.

---

## Installation

We recommend using **Python 3.11** and a clean Conda environment:

```bash
conda create -n embqa python=3.11
conda activate embqa
