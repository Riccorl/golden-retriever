#!/bin/bash

bash scripts/train.sh \
    --config-path conf/pretrain_iterable_in_batch.yaml \
    -l thenlper/gte-base \
    --print \
    --wandb golden-retriever-blink \
    -m 24000 \
    "model=golden_retriever" \
    "model_name=gte-base-blink-inbatch-first1M-random-hnprob-0.2-hf" \
    "data.shared_params.passage_batch_size=400" \
    "++train.pl_trainer.num_sanity_val_steps=0" \
    "++data.datamodule.datasets.train.prefetch=True" \
    "train.callbacks.hard_negatives_callback.add_with_probability=0.2" \
    "data.datamodule.datasets.train.subsample_strategy=random" \
    "data.shared_params.passages_path=/home/ric/projects/golden-retriever-v2/data/dpr-like/el/definitions.txt" \
    "data.datamodule.datasets.train.path=['/media/ssd/ric/data/golden/EL/blink/window_32_tokens/random_1M/dpr-like/first_1M.jsonl']" \
    "data.datamodule.datasets.val.0.path=['/media/ssd/ric/data/golden/EL/blink/window_32_tokens/random_1M/dpr-like/val.jsonl']" \
    "data.datamodule.datasets.test.0.path=['/media/ssd/ric/data/golden/EL/blink/window_32_tokens/random_1M/dpr-like/val.jsonl']"
