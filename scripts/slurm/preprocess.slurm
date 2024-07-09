#!/bin/bash
#SBATCH --job-name=preprocess-data            # Job name
#SBATCH -o logs/preprocess-data.out       # Name of stdout output file
#SBATCH -e logs/preprocess-data.err       # Name of stderr error file

#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --time 4:00:00          # format: HH:MM:SS

#SBATCH -A IscrC_DELEE
#SBATCH -p lrd_all_serial

source /leonardo_work/IscrC_DELEE/python-envs/golden-retriever-dist/bin/activate

python scripts/preprocess_data_to_mds.py /leonardo_work/IscrC_DELEE/entity-rag/data/aida+ood+zelda_contexts_en.similarity.goldenindex.06-july.top-10.dpr.jsonl --num-workers 8