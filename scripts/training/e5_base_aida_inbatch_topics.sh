#!/bin/bash

bash scripts/train.sh \
    --config-path conf/finetune_iterable_in_batch.yaml \
    -l intfloat/e5-base-v2 \
    --print \
    --wandb golden-retriever-aida \
    "data=aida_dataset" \
    "model_name=e5-base-aida-inbatch-topics" \
    "train.pretrain_ckpt_path=/root/golden-retriever-2/experiments/e5-base-mosaico-inbatch-first1M-wsd-random-hnprob0.2/2023-07-02/17-44-33/wandb/latest-run/files/checkpoints/checkpoint-validate_recall@100_0.9622-epoch_12.ckpt" \
    "data.shared_params.use_topics=True" \
    "data.shared_params.context_batch_size=400" \
    "data.shared_params.contexts_path=data/dpr-like/el/definitions.txt" \
    "data.datamodule.datasets.train.path=['data/dpr-like/el/aida_32_tokens_topic/train.jsonl']" \
    "data.datamodule.datasets.val.0.path=['data/dpr-like/el/aida_32_tokens_topic/val.jsonl']" \
    "data.datamodule.datasets.test.0.path=['data/dpr-like/el/aida_32_tokens_topic/test.jsonl']"
