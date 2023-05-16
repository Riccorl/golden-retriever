#!/bin/bash

# setup conda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE"/bin/activate golden

# add candidates to all files in a folder
for FILEPATH in /media/data/EL/genre/edo/32_window/splits/*.jsonl; do
  echo "Retrieving candidates for $FILEPATH"
  # put the output in a folder called "candidates" in the same directory
  OUTPUT_DIR=$(dirname "$FILEPATH")/mpnet-1encoder-topics-rebel7epochs-32words
  OUTPUT_FILE=$(basename "$FILEPATH")
  OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_FILE%.*}_candidates.jsonl"
  echo "Output will be saved to $OUTPUT_FILE"
  python scripts/data/aida/add_candidates.py \
    --retriever_name_or_path retrievers/mpnet-1encoder-topics-rebel7epochs-32words \
    --input_path "$FILEPATH" \
    --output_path "$OUTPUT_FILE" \
    --precision 16 \
    --batch_size 256
done
