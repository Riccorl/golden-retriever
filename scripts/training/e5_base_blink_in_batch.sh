#!/bin/bash

bash scripts/train.sh \
    --config-path conf/pretrain_iterable_in_batch.yaml \
    -l intfloat/e5-base-v2 \
    --print \
    --wandb golden-retriever-blink \
    "model_name=e5-base-blink-inbatch-first1M" \
    "data.shared_params.context_batch_size=400" \
    "++train.pl_trainer.num_sanity_val_steps=0" \
    "++data.datamodule.datasets.train.prefetch=True" \
    "data.shared_params.contexts_path=data/dpr-like/el/definitions.txt" \
    "data.datamodule.datasets.train.subsample_strategy=random" \
    "data.datamodule.datasets.train.path=['/media/data/EL/blink/window_32_tokens/random_1M/dpr-like/first_1M.jsonl']" \
    "data.datamodule.datasets.val.0.path=['data/dpr-like/el/blink/val.jsonl']" \
    "data.datamodule.datasets.test.0.path=['data/dpr-like/el/blink/val.jsonl']"
