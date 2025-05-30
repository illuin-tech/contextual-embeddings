# Contextual Embeddings

This is the repo that holds all the code of the project on training a contextual embedding model.

## Installation

```bash
pip install -e .
```

## Usage

Example configurations can be found in the `configs` directory. To run a training job, use the following command:

```bash
accelerate launch scripts/training/training.py scripts/configs/examples/modernbert.yaml
```