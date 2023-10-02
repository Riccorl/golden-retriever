#!/bin/bash

bash scripts/train.sh \
    --config-path conf/finetune_iterable_in_batch.yaml \
    -l riccorl/golden-retriever-base-blink-before-hf \
    --print \
    --wandb golden-retriever-aida-abl \
    -m 24000 \
    "data=aida_dataset" \
    "model.pl_module.model.question_encoder=intfloat/e5-base-v2" \
    "model_name=e5-base-aida-inbatch-topics-hf-dualencoder-passage-from-blink-beforehf" \
    "data.shared_params.use_topics=True" \
    "data.shared_params.passage_batch_size=400" \
    "data.shared_params.passages_path=/root/golden-retriever-v2/data/dpr-like/el/definitions.txt" \
    "data.datamodule.datasets.train.path=['data/dpr-like/el/aida_32_tokens_topic/train.jsonl']" \
    "data.datamodule.datasets.val.0.path=['data/dpr-like/el/aida_32_tokens_topic/val.jsonl']" \
    "data.datamodule.datasets.test.0.path=['data/dpr-like/el/aida_32_tokens_topic/test.jsonl']"

    # "model.pl_module.model.question_encoder=/root/golden-retriever-v2/experiments/e5-base-blink-inbatch-first1M-random-hnprob-0.2-hf/2023-09-12/09-35-03/wandb/run-20230912_093521-24uqnphn/files/retriever/question_encoder" \
    # "model.pl_module.model.passage_encoder=/root/golden-retriever-v2/experiments/e5-base-blink-inbatch-first1M-random-hnprob-0.2-hf/2023-09-12/09-35-03/wandb/run-20230912_093521-24uqnphn/files/retriever/passage_encoder" \
    # -l riccorl/golden-retriever-base-blink-before-hf \
    # "train.pretrain_ckpt_path=/home/ric/projects/golden-retriever-v2/experiments/e5-base-blink-inbatch-first1M-random-hnprob-0.2/2023-07-11/20-51-29/wandb/run-20230711_205151-v8saqfvh/files/checkpoints/checkpoint-validate_recall@100_0.9707-epoch_79.ckpt" \
    # "train.pretrain_ckpt_path=/home/ric/projects/golden-retriever-v2/experiments/e5-base-blink-inbatch-first1M-random-hnprob-0.2/2023-07-11/20-51-29/wandb/run-20230711_205151-v8saqfvh/files/checkpoints/checkpoint-validate_recall@100_0.9707-epoch_79.ckpt" \
