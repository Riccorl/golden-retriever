#!/bin/bash

#!/bin/bash
#SBATCH --job-name=preprocess-data            # Job name
#SBATCH -o /leonardo_work/IscrC_DELEE/golden-retriever-dist/training_logs/entity-rag/e5-base-v2-index-12-july-self-target/e5-base-v2-index-12-july-self-target.out       # Name of stdout output file
#SBATCH -e /leonardo_work/IscrC_DELEE/golden-retriever-dist/training_logs/entity-rag/e5-base-v2-index-12-july-self-target/e5-base-v2-index-12-july-self-target.err       # Name of stderr error file

#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=4     # number of tasks per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time 24:00:00          # format: HH:MM:SS

#SBATCH -A IscrC_DELEE
#SBATCH -p boost_usr_prod #lrd_all_serial


# MODULES="cuda/12.1"
# CONFIG_PATH=conf/entity_rag_self_base.yaml

# # module load $MODULES
# source /leonardo_work/IscrC_DELEE/python-envs/golden-retriever-dist/bin/activate

# export HF_HOME=/leonardo_work/IscrC_DELEE/hf_cache
# export HF_DATASETS_CACHE=/leonardo_work/IscrC_DELEE/hf_cache
# export HUGGINGFACE_HUB_CACHE=/leonardo_work/IscrC_DELEE/hf_cache
# export WANDB_MODE=offline
# # get Huggingface token from python
# export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")

# export NCCL_IB_SL=1
# export UCX_IB_SL=1
# export NVSHMEM_IB_SL=1
# export NVSHMEM_DISABLE_NCCL=1

golden-retriever train $CONFIG_PATH

# exit from ssh session
# exit
