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

# How to use

Install the library from [PyPi](https://pypi.org/project/goldenretriever-core/):

```bash
pip install goldenretriever-core
```

or from source:

```bash
git clone https://github.com/Riccorl/golden-retriever.git
cd golden-retriever
pip install -e .
```

### FAISS

Install with optional dependencies for [FAISS](https://github.com/facebookresearch/faiss)

FAISS pypi package is only available for CPU. If you want to use GPU, you need to install it from source or use the conda package.

For CPU:

```bash
pip install goldenretriever-core[faiss]
```

For GPU:

```bash
conda create -n goldenretriever python=3.11
conda activate goldenretriever

# install pytorch
conda install -y pytorch=2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# GPU
conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0
# or GPU with NVIDIA RAFT
conda install -y -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-raft=1.8.0

pip install goldenretriever-core
```

# Usage

Golden Retriever is built on top of PyTorch Lightning and Hydra. To run an experiment, you need to create a configuration file and pass 
it to the `golden-retriever` command. Few examples are provided in the `conf` folder.

## Training

Here a simple example on how to train a DPR-like Retriever on the NQ dataset.
First download the dataset from [DPR](https://github.com/facebookresearch/DPR?tab=readme-ov-file#retriever-input-data-format). The run the following code:

```bash
golden-retriever train conf/nq-dpr.yaml
```

## Evaluation

TODO

### Distributed environment

Golden Retriever supports distributed training. For the moment, it is only possible to train on a single node with multiple GPUs and without model sharding, i.e.
only DDP and FSDP with `NO_SHARD` strategy are supported.

To run a distributed training, just add the following keys to the configuration file:

```yaml
devices: 4  # number of GPUs
# strategy: "ddp_find_unused_parameters_true"  # DDP
# FSDP with NO_SHARD
strategy:
  _target_: lightning.pytorch.strategies.FSDPStrategy
  sharding_strategy: "NO_SHARD"
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
```

### Index data

The document to index can be either a jsonl file or a tsv file similar to
[DPR](https://github.com/facebookresearch/DPR):

- `jsonl`: each line is a json object with the following keys: `id`, `text`, `metadata`
- `tsv`: each line is a tab-separated string with the `id` and `text` column,
  followed by any other column that will be stored in the `metadata` field

jsonl example:

```json lines
{
  "id": "...",
  "text": "...",
  "metadata": ["{...}"]
},
...
```

tsv example:

```tsv
id \t text \t any other column
...
```
