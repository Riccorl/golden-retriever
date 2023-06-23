#!/bin/bash

bash scripts/train.sh \
    --config-path conf/finetune_iterable_in_batch.yaml \
    -l intfloat/e5-base-v2 \
    --print \
    --wandb golden-retriever-aida \
    -o \
    "model_name=e5-base-aida-inbatch-topics" \
    "data.shared_params.max_contexts_per_batch=400" \
    "data.shared_params.use_topics=True" \
    "data.shared_params.prefetch_batches=False" \
    "data.shared_params.contexts_path=data/dpr-like/el/definitions_only_data.txt" \
    "data.datamodule.datasets.train.path=['data/dpr-like/el/aida_32_tokens_topic/train.jsonl']" \
    "data.datamodule.datasets.val.0.path=['data/dpr-like/el/aida_32_tokens_topic/val.jsonl']" \
    "data.datamodule.datasets.test.0.path=['data/dpr-like/el/aida_32_tokens_topic/test.jsonl']"
