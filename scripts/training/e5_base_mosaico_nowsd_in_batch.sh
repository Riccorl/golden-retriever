#!/bin/bash

bash scripts/train.sh \
    --config-path conf/pretrain_iterable_in_batch.yaml \
    -l intfloat/e5-base-v2 \
    --print \
    --wandb golden-retriever-mosaico \
    "model_name=e5-base-mosaico-inbatch-first1M-nowsd-random-hnprob0.2" \
    "data.shared_params.passage_batch_size=400" \
    "++train.pl_trainer.num_sanity_val_steps=0" \
    "++data.datamodule.datasets.train.prefetch=True" \
    "train.callbacks.hard_negatives_callback.add_with_probability=0.2" \
    "data.datamodule.datasets.train.subsample_strategy=random" \
    "data.shared_params.passages_path=data/dpr-like/el/definitions.txt" \
    "data.datamodule.datasets.train.path=['/media/ssd/ric/data/golden/EL/mosaico/window_32_tokens/nowsd/dpr/first_1M.jsonl']" \
    "data.datamodule.datasets.val.0.path=['/media/ssd/ric/data/golden/EL/mosaico/window_32_tokens/dpr/val.jsonl']" \
    "data.datamodule.datasets.test.0.path=['/media/ssd/ric/data/golden/EL/mosaico/window_32_tokens/dpr/val.jsonl']"
