#!/bin/bash

QUESTION_ENCODER=/root/golden-retriever-v2/experiments/e5-base-aida-inbatch-hf-dualencoder-nohard/2023-09-18/20-48-48/wandb/latest-run/files/retriever/question_encoder \
PASSAGE_ENCODE=/root/golden-retriever-v2/experiments/e5-base-aida-inbatch-hf-dualencoder-nohard/2023-09-18/20-48-48/wandb/latest-run/files/retriever/passage_encoder \
DOCUMENT_INDEX=/root/golden-retriever-v2/experiments/e5-base-aida-inbatch-hf-dualencoder-nohard/2023-09-18/20-48-48/wandb/latest-run/files/retriever/document_index \
WINDOW_BATCH_SIZE=64 \
PRECISION=16 \
INDEX_PRECISION=16 \
DEVICE=cuda \
serve run goldenretriever.serve.server.ray:server

# PASSAGE_ENCODER=/root/golden-retriever-v2/experiments/e5-base-aida-inbatch-hf-dualencoder/2023-09-18/08-28-25/wandb/run-20230918_082845-7fnfdjq0/files/retriever/passage_encoder \