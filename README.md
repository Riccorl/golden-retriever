<h1 align="center">
  🦮 Golden Retriever
</h1>

<p align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?logo=pytorch"></a>
  <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet"></a>
  <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg"></a>
  <a href="https://github.dev/Riccorl/golden-retriever"><img alt="vscode" src="https://img.shields.io/badge/preview%20in-vscode.dev-blue"></a>
</p>
<p align="center">
  <a href="https://github.com/Riccorl/golden-retriever/releases"><img alt="release" src="https://img.shields.io/github/v/release/Riccorl/golden-retriever"></a>
  <a href="https://github.com/Riccorl/golden-retriever/actions/workflows/python-publish-pypi.yml"><img alt="gh-status" src="https://github.com/Riccorl/golden-retriever/actions/workflows/python-publish-pypi.yml/badge.svg"></a>

</p>

# WIP: distributed-compatible codebase

A distributed-compatible codebase is under development. Check the `distributed` [branch](https://github.com/Riccorl/golden-retriever/tree/distributed) for the latest updates.

# How to use

Install the library from [PyPI](https://pypi.org/project/goldenretriever-core/):

```bash
pip install goldenretriever-core
```

or from source:

```bash
git clone https://github.com/Riccorl/golden-retriever.git
cd golden-retriever
pip install -e .
```

# Usage

## How to run an experiment

### Training

Here a simple example on how to train a DPR-like Retriever on the NQ dataset.
First download the dataset from [DPR](https://github.com/facebookresearch/DPR?tab=readme-ov-file#retriever-input-data-format). The run the following code:

```python
from goldenretriever.trainer import Trainer
from goldenretriever import GoldenRetriever
from goldenretriever.data.datasets import InBatchNegativesDataset

# create a retriever
retriever = GoldenRetriever(
    question_encoder="intfloat/e5-small-v2",
    passage_encoder="intfloat/e5-small-v2"
)

# create a dataset
train_dataset = InBatchNegativesDataset(
    name="webq_train",
    path="path/to/webq_train.json",
    tokenizer=retriever.question_tokenizer,
    question_batch_size=64,
    passage_batch_size=400,
    max_passage_length=64,
    shuffle=True,
)
val_dataset = InBatchNegativesDataset(
    name="webq_dev",
    path="path/to/webq_dev.json",
    tokenizer=retriever.question_tokenizer,
    question_batch_size=64,
    passage_batch_size=400,
    max_passage_length=64,
)

trainer = Trainer(
    retriever=retriever,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    max_steps=25_000,
    wandb_online_mode=True,
    wandb_project_name="golden-retriever-dpr",
    wandb_experiment_name="e5-small-webq",
    max_hard_negatives_to_mine=5,
)

# start training
trainer.train()
```

### Evaluation

```python
from goldenretriever.trainer import Trainer
from goldenretriever import GoldenRetriever
from goldenretriever.data.datasets import InBatchNegativesDataset

retriever = GoldenRetriever(
  question_encoder="",
  document_index="",
  device="cuda",
  precision="16",
)

test_dataset = InBatchNegativesDataset(
  name="test",
  path="",
  tokenizer=retriever.question_tokenizer,
  question_batch_size=64,
  passage_batch_size=400,
  max_passage_length=64,
)

trainer = Trainer(
  retriever=retriever,
  test_dataset=test_dataset,
  log_to_wandb=False,
  top_k=[20, 100]
)

trainer.test()
```

## Inference

```python
from goldenretriever import GoldenRetriever

retriever = GoldenRetriever(
    question_encoder="path/to/question/encoder",
    passage_encoder="path/to/passage/encoder",
    document_index="path/to/document/index"
)

# retrieve documents
retriever.retrieve("What is the capital of France?", k=5)
```

## Data format

### Input data

The retriever expects a jsonl file similar to [DPR](https://github.com/facebookresearch/DPR):

```json lines
[
  {
  "question": "....",
  "answers": ["...", "...", "..."],
  "positive_ctxs": [{
    "title": "...",
    "text": "...."
  }],
  "negative_ctxs": ["..."],
  "hard_negative_ctxs": ["..."]
  },
  ...
]
```

### Index data

The document to index can be either a jsonl file or a tsv file similar to
[DPR](https://github.com/facebookresearch/DPR):

- `jsonl`: each line is a json object with the following keys: `id`, `text`, `metadata`
- `tsv`: each line is a tab-separated string with the `id` and `text` column,
  followed by any other column that will be stored in the `metadata` field

jsonl example:

```json lines
[
  {
    "id": "...",
    "text": "...",
    "metadata": ["{...}"]
  },
  ...
]
```

tsv example:

```tsv
id \t text \t any other column
...
```
