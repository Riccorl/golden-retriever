#!/bin/bash

bash scripts/train.sh \
    --config-path conf/pretrain_iterable_in_batch.yaml \
    -l intfloat/e5-small-v2 \
    --print \
    --wandb golden-retriever-mosaico \
    "model_name=e5-small-mosaico-inbatch-first1M-wsd" \
    "data.shared_params.context_batch_size=400" \
    "data.shared_params.contexts_path=data/dpr-like/el/definitions.txt" \
    "data.datamodule.datasets.train.subsample_strategy=in_order" \
    "data.datamodule.datasets.train.path=['data/dpr-like/el/mosaico/wsd/first_1M.jsonl']" \
    "data.datamodule.datasets.val.0.path=['data/dpr-like/el/mosaico/wsd/val.jsonl']" \
    "data.datamodule.datasets.test.0.path=['data/dpr-like/el/mosaico/wsd/val.jsonl']"
    # "train.callbacks.hard_negatives_callback=null"
