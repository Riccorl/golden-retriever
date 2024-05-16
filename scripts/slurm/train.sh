#!/bin/bash
#SBATCH --job-name=minestral-1B-100B_it-100B_en-cx-04032024-train            # Job name
#SBATCH -o logs/minestral-1B-100B_it-100B_en-cx-04032024/train-test-job.out       # Name of stdout output file
#SBATCH -e logs/minestral-1B-100B_it-100B_en-cx-04032024/train-test-job.err       # Name of stderr error file
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=4             # number of tasks per node
#SBATCH --cpus-per-task=8              # number of threads per task
#SBATCH --time 24:00:00                  # format: HH:MM:SS
#SBATCH --gres=gpu:2                    # number of gpus per node

#SBATCH -A IscrC_MEL
#SBATCH -p boost_usr_prod

module load profile/deeplrn cuda/12.1

export GOLDENRETRIEVER_CACHE_DIR=$SCRATCH/golden_retriever_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# get Huggingface token from python
export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

composer scripts/train/train.py scripts/train/yamls/pretrain/minestral-1B-100B_it-100B_en-cx-04032024.yaml
