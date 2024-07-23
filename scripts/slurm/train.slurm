#!/bin/bash

MODULES="cuda/12.1"
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
