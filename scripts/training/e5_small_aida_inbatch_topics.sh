#!/bin/bash

bash scripts/train.sh \
    --config-path conf/finetune_iterable_in_batch.yaml \
    -l intfloat/e5-small-v2 \
    --print \
    --wandb golden-retriever-aida \
    "model_name=e5-small-aida-inbatch-topics" \
    "data.shared_params.use_topics=True" \
    "data.shared_params.context_batch_size=400" \
    "data.shared_params.contexts_path=data/dpr-like/el/definitions.txt" \
    "data.datamodule.datasets.train.path=['data/dpr-like/el/aida_32_tokens_topic/train.jsonl']" \
    "data.datamodule.datasets.val.0.path=['data/dpr-like/el/aida_32_tokens_topic/val.jsonl']" \
    "data.datamodule.datasets.test.0.path=['data/dpr-like/el/aida_32_tokens_topic/test.jsonl']"
