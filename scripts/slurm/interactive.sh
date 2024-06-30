NUM_GPUS=$1
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=1
fi

export TORCH_NCCL_USE_COMM_NONBLOCKING=1

export NCCL_IB_SL=1
export UCX_IB_SL=1
export NVSHMEM_IB_SL=1
export NVSHMEM_DISABLE_NCCL=1

srun -N 1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 --time 2:00:00 -p boost_usr_prod -A IscrC_MEL --pty /bin/bash
