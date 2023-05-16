#!/bin/bash

# setup conda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE"/bin/activate golden


# add candidates to all files in a folder
 for FILEPATH in /media/data/EL/genre/32_window/splits/*.jsonl; do
   echo "Retrieving candidates for $FILEPATH"
   echo "Output will be saved to ${FILEPATH%.jsonl}_candidates.jsonl"
   python scripts/data/aida/add_candidates.py \
     --retriever_name_or_path retrievers/mpnet-1encoder-topics-rebel-32words/ \
     --input_path "$FILEPATH" \
     --output_path "${FILEPATH%.jsonl}"_candidates.jsonl \
     --precision 16 \
     --batch_size 256
 done
