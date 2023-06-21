#!/bin/bash

bash scripts/train.sh \
    --config-path conf/finetune_iterable_in_batch.yaml \
    -l intfloat/e5-base-v2 \
    --print \
    --wandb golden-retriever-aida \
    "model_name=e5-base-aida-inbatch" \
    "data.shared_params.max_contexts_per_batch=400" \
    "data.shared_params.contexts_path=data/dpr-like/el/definitions.txt" \
    "data.datamodule.datasets.train.path=['data/dpr-like/el/aida_32_tokens/train.jsonl']" \
    "data.datamodule.datasets.val.0.path=['data/dpr-like/el/aida_32_tokens/val.jsonl']" \
    "data.datamodule.datasets.test.0.path=['data/dpr-like/el/aida_32_tokens/test.jsonl']"
