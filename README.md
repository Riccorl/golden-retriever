<h1 align="center">
  🦮 Golden Retriever
</h1>

<p align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?logo=pytorch"></a>
  <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet"></a>
  <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg"></a>
</p>

# How to use

Install the library from [PyPI]():

```bash
pip install goldenretriever
```

or from source:

```bash
git clone https://github.com/Riccorl/goldenretriever.git
cd goldenretriever
pip install -e .
```

# Usage

## How to run an experiment

### Training

Here a simple example on how to train a DPR-like Retriever on the NQ dataset.
First download the dataset from (DPR)[]. The run the following code:

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
    name="nq_train",
    path="path/to/nq_train.json",
    tokenizer=retriever.question_tokenizer,
    question_batch_size=64,
    passage_batch_size=400,
    max_passage_length=64,
    shuffle=True,
)
val_dataset = InBatchNegativesDataset(
    name="nq_dev",
    path="path/to/nq_dev.json",
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
    wandb_project_name="golden-retriever",
    wandb_experiment_name="e5-small-nq",
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

# logger.info("Loading document index")
# document_index = InMemoryDocumentIndex(
#     documents=documents,
#     # metadata_fields=["title"],
#     # separator=" <title> ",
#     device="cuda",
#     precision="16",
# )
# retriever.document_index = document_index

trainer = Trainer(
    retriever=retriever,
    test_dataset=test_dataset,
    log_to_wandb=False,
    top_k=[20, 100]
)

# trainer.train()
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

