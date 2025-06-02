# ConTEB: Context is Gold to find the Gold Passage: Evaluating and Training Contextual Document Embeddings

This repository contains all training and inference code released with our preprint [*Context is Gold to find the Gold Passage: Evaluating and Training Contextual Document Embeddings*](https://arxiv.org/abs/2505.24782).


[![arXiv](https://img.shields.io/badge/arXiv-2505.24782-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2505.24782)

<img src="https://cdn-uploads.huggingface.co/production/uploads/60f2e021adf471cbdf8bb660/jq_zYRy23bOZ9qey3VY4v.png" width="800">


## Installation

```bash
pip install -e .
```

## Training

Example configurations can be found in the `configs` directory. To run a training job, use the following command:

```bash
accelerate launch scripts/training/training.py scripts/configs/examples/modernbert.yaml
```

## Evaluation

Code for evaluation can be found in the [ConTEB](https://github.com/illuin-tech/conteb) repository.


### Abstract

A limitation of modern document retrieval embedding methods is that they typically encode passages (chunks) from the same documents independently, often overlooking crucial contextual information from the rest of the document that could greatly improve individual chunk representations.

In this work, we introduce *ConTEB* (Context-aware Text Embedding Benchmark), a benchmark designed to evaluate retrieval models on their ability to leverage document-wide context. Our results show that state-of-the-art embedding models struggle in retrieval scenarios where context is required. To address this limitation, we propose *InSeNT* (In-sequence Negative Training), a novel contrastive post-training approach which combined with \textit{late chunking} pooling enhances contextual representation learning while preserving computational efficiency. Our method significantly improves retrieval quality on *ConTEB* without sacrificing base model performance. 
We further find chunks embedded with our method are more robust to suboptimal chunking strategies and larger retrieval corpus sizes.
We open-source all artifacts here and at https://github.com/illuin-tech/contextual-embeddings.

## Ressources

- [*HuggingFace Project Page*](https://huggingface.co/illuin-conteb): The HF page centralizing everything!
- [*(Model) ModernBERT*](TODO): The Contextualized ModernBERT bi-encoder trained with InSENT loss and Late Chunking
- [*(Model) ModernColBERT*](TODO): The Contextualized ModernColBERT trained with InSENT loss and Late Chunking
- [*Leaderboard*](TODO): Coming Soon
- [*(Data) ConTEB Benchmark Datasets*](TODO):
- [*(Code) Contextual Document Engine*](https://github.com/illuin-tech/contextual-embeddings): The code used to train and run inference with our architecture.
- [*(Code) ConTEB Benchmarkk*](https://github.com/illuin-tech/conteb): A Python package/CLI tool to evaluate document retrieval systems on the ConTEB benchmark.
- [*Preprint*](https://arxiv.org/abs/2505.24782): The paper with all details !
- [*Blog*](https://huggingface.co/XXX): TODO

## Contact

- Manuel Faysse: manuel.faysse@illuin.tech
- Max Conti: max.conti@illuin.tech

## Citation

If you use any datasets or models from this organization in your research, please cite the original dataset as follows:

```latex
@misc{conti2025contextgoldgoldpassage,
      title={Context is Gold to find the Gold Passage: Evaluating and Training Contextual Document Embeddings}, 
      author={Max Conti and Manuel Faysse and Gautier Viaud and Antoine Bosselut and Céline Hudelot and Pierre Colombo},
      year={2025},
      eprint={2505.24782},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2505.24782}, 
}
```

## Acknowledgments

This work is partially supported by [ILLUIN Technology](https://www.illuin.tech/), and by a grant from ANRT France.
This work was performed using HPC resources from the GENCI Jeanzay supercomputer with grant AD011016393.

