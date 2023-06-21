#!/bin/bash

bash scripts/train.sh \
    --config-path conf/pretrain_iterable_in_batch.yaml \
    -l intfloat/e5-base-v2 \
    --print \
    --wandb golden-retriever-mosaico \
    "model_name=e5-base-mosaico-inbatch-first1M-wsd" \
    "data.shared_params.use_topics=False" \
    "data.shared_params.max_contexts_per_batch=400" \
    "data.shared_params.prefetch_batches=False" \
    "data.shared_params.contexts_path=data/dpr-like/el/definitions.txt" \
    "data.datamodule.datasets.train.path=['data/dpr-like/el/mosaico/wsd/first_1M.jsonl']" \
    "data.datamodule.datasets.val.0.path=['data/dpr-like/el/mosaico/wsd/val.jsonl']" \
    "data.datamodule.datasets.test.0.path=['data/dpr-like/el/mosaico/wsd/val.jsonl']"
