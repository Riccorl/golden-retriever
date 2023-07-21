#!/bin/bash

bash scripts/train.sh \
    --config-path conf/finetune_iterable_in_batch.yaml \
    -l riccorl/e5-base-v2-blink \
    --print \
    --wandb golden-retriever-aida \
    -m 24000 \
    "train.pretrain_ckpt_path=/home/ric/projects/golden-retriever-v2/experiments/e5-base-blink-inbatch-first1M-random-hnprob-0.2/2023-07-11/20-51-29/wandb/run-20230711_205151-v8saqfvh/files/checkpoints/checkpoint-validate_recall@100_0.9707-epoch_79.ckpt" \
    "data=aida_dataset" \
    "model_name=e5-base-aida-inbatch-topics-from-blink-reset-each-epoch" \
    "data.shared_params.use_topics=True" \
    "data.shared_params.passage_batch_size=400" \
    "data.shared_params.passages_path=/home/ric/projects/golden-retriever-v2/data/dpr-like/el/definitions.txt" \
    "data.datamodule.datasets.train.path=['data/dpr-like/el/aida_32_tokens_topic/train.jsonl']" \
    "data.datamodule.datasets.val.0.path=['data/dpr-like/el/aida_32_tokens_topic/val.jsonl']" \
    "data.datamodule.datasets.test.0.path=['data/dpr-like/el/aida_32_tokens_topic/test.jsonl']"

    # "train.pretrain_ckpt_path=/home/ric/projects/golden-retriever-v2/experiments/e5-base-blink-inbatch-first1M-random-hnprob-0.2/2023-07-11/20-51-29/wandb/run-20230711_205151-v8saqfvh/files/checkpoints/checkpoint-validate_recall@100_0.9707-epoch_79.ckpt" \