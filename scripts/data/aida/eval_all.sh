#!/bin/bash

EXP_FOLDER="e5-base-aida-inbatch-hf-dualencoder-nohard"


# testa
echo "testa"
python \
scripts/data/aida/add_candidates_rest.py \
--input_path data/aida/window_32_tokens/testa_windowed.jsonl \
--output_path data/aida/window_32_tokens/abl/"$EXP_FOLDER"/testa_windowed_candidates.jsonl  \
--batch_size 128 \
--topics

# testb
echo "testb"
python \
scripts/data/aida/add_candidates_rest.py \
--input_path data/aida/window_32_tokens/testb_windowed.jsonl \
--output_path data/aida/window_32_tokens/abl/"$EXP_FOLDER"/testb_windowed_candidates.jsonl  \
--batch_size 128 \
--topics

# msnbc
echo "msnbc"
python \
scripts/data/aida/add_candidates_rest.py \
--input_path data/aida/ood/32_windows/msnbc_windowed.jsonl \
--output_path data/aida/window_32_tokens/abl/"$EXP_FOLDER"/msnbc_windowed_candidates.jsonl  \
--batch_size 128 \
--topics

# train
echo "train"
python \
scripts/data/aida/add_candidates_rest.py \
--input_path data/aida/window_32_tokens/train_windowed.jsonl \
--output_path data/aida/window_32_tokens/abl/"$EXP_FOLDER"/train_windowed_candidates.jsonl  \
--batch_size 128 \
--topics

