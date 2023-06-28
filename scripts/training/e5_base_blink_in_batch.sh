#!/bin/bash

bash scripts/train.sh \
    --config-path conf/pretrain_iterable_in_batch.yaml \
    -l intfloat/e5-base-v2 \
    --print \
    --wandb golden-retriever-blink \
    "model_name=e5-base-blink-inbatch-first1M" \
    "data.shared_params.context_batch_size=400" \
    "++data.datamodule.datasets.train.prefetch=False" \
    "data.shared_params.contexts_path=data/dpr-like/el/definitions.txt" \
    "data.datamodule.datasets.train.subsample_strategy=in_order" \
    "data.datamodule.datasets.train.path=['data/dpr-like/el/blink/first_1M.jsonl']" \
    "data.datamodule.datasets.val.0.path=['data/dpr-like/el/blink/val.jsonl']" \
    "data.datamodule.datasets.test.0.path=['data/dpr-like/el/blink/val.jsonl']"
