#!/bin/bash

USAGE="Usage: train.sh [OPTIONS]

This script is used to train a model with various parameters.

Options:
  -h                Show this help message and exit
  -c CONFIG_PATH    Path to the configuration file
  -v PYTHON_ENV     Python environment to use
  -n NODES          Number of nodes to use
  -l LOGS_PATH      Path to store logs
  -m MODULES        Modules to load
  -t TIME           Time for the job
  -a ACCOUNT        Account to use for the job
  -p PARTITION      Partition to use for the job
  -k CHAIN          Number of job to chain
  -j JOB_NAME       Name of the job
  -o STD_OUT        Path to standard output file
  -e STD_ERR        Path to standard error file
  -x                Use exclusive node
  -t TRAINING_SCRIPT Path to the training script
  -g GPU_PER_NODE   Number of GPUs per node
  -i INTERACTIVE    Run the job interactively

Invalid options will show this help message and exit.
"

# check for named params
#while [ $OPTIND -le "$#" ]; do
while getopts ":hc:v:n:l:m:t:a:p:j:e:o:xt:g:ik:" opt; do
    case $opt in
    h)
        printf "%s$USAGE" && exit 0
        ;;
    c)
        CONFIG_PATH="$OPTARG"
        ;;
    v)
        PYTHON_ENV="$OPTARG"
        ;;
    n)
        NODES="$OPTARG"
        ;;
    l)
        LOGS_PATH="$OPTARG"
        ;;
    m)
        MODULES="$OPTARG"
        ;;
    t)
        TIME="$OPTARG"
        ;;
    a)
        ACCOUNT="$OPTARG"
        ;;
    p)
        PARTITION="$OPTARG"
        ;;
    k)
        CHAIN="$OPTARG"
        ;;
    j)
        JOB_NAME="$OPTARG"
        ;;
    o)
        STD_OUT="$OPTARG"
        ;;
    e)
        STD_ERR="$OPTARG"
        ;;
    x)
        EXCLUSIVE="TRUE"
        ;;
    t)
        TRAINING_SCRIPT="$OPTARG"
        ;;
    g)
        GPU_PER_NODE="$OPTARG"
        ;;
    i)
        INTERACTIVE="TRUE"
        ;;
    \?)
        echo "Invalid option -$OPTARG" >&2 && echo "$USAGE" && exit 0
        ;;
    esac
done

if [ -z "$CONFIG_PATH" ]; then
    echo "CONFIG_PATH is not set. To set it, use the -c flag" && exit 1
fi

if [ -z "$PYTHON_ENV" ]; then
    PYTHON_ENV=/leonardo_scratch/large/userexternal/rorland1/python-envs/golden-dist-venv/bin/activate
fi

if [ -z "$NODES" ]; then
    NODES=1
fi

if [ -z "$MODULES" ]; then
    MODULES="profile/deeplrn cuda/12.1"
fi

if [ -z "$TIME" ]; then
    TIME=24:00:00
fi

if [ -z "$ACCOUNT" ]; then
    ACCOUNT=IscrB_medit
fi

if [ -z "$PARTITION" ]; then
    PARTITION=boost_usr_prod
fi

if [ -z "$CHAIN" ]; then
    CHAIN=1
fi

if [ -z "$JOB_NAME" ]; then
    # it not interactive, raise an error
    if [ "$INTERACTIVE" = "FALSE" ]; then
        echo "JOB_NAME is not set. To set it, use the -j flag" && exit 1
    fi
fi

if [ -z "$LOGS_PATH" ]; then
    # default logs path is in SCRATCH folder
    LOGS_PATH="$SCRATCH/golden-retriever-dist/training_logs"
    # if logs path does not exist, create it
    if [ ! -d "$LOGS_PATH/$JOB_NAME" ]; then
        mkdir -p "$LOGS_PATH/$JOB_NAME"
    fi
fi

if [ -z "$STD_OUT" ]; then
    # extract the last part of the JOB_NAME if it is a file path
    JOB_NAME_FILE_NAME=$(basename $JOB_NAME)
    STD_OUT="$LOGS_PATH/$JOB_NAME/$JOB_NAME_FILE_NAME.out"
fi

if [ -z "$STD_ERR" ]; then
    # extract the last part of the JOB_NAME if it is a file path
    JOB_NAME_FILE_NAME=$(basename $JOB_NAME)
    STD_ERR="$LOGS_PATH/$JOB_NAME/$JOB_NAME_FILE_NAME.err"
fi

if [ -z "$EXCLUSIVE" ]; then
    EXCLUSIVE=""
else
    EXCLUSIVE="--exclusive"
fi

if [ -z "$GPU_PER_NODE" ]; then
    GPU_PER_NODE=1
fi

if [ -z "$INTERACTIVE" ]; then
    INTERACTIVE="FALSE"
fi

module load $MODULES
source "$PYTHON_ENV"

# get the absolute path of the current directory
CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CURRENT_DIR
cd $CURRENT_DIR

# check whether the configuration file is a relative path
if [[ ${CONFIG_PATH:0:1} == '/' ]]; then
    CONFIG_PATH=$CONFIG_PATH
else
    # provide error

fi

# if NODES is 1, then we don't need all this shit'
export NODES
export GPU_PER_NODE
# if [ $NODES -gt 1 ]; then
#     # export NPROCS=$GPU_PER_NODE # number of GPUs per node
#     # export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#     # export MASTER_PORT=11111
#     # export WORLD_SIZE=$(($SLURM_NNODES * $NPROCS))
# fi

# check if $SCRATCH/hf_cache exists
if [ ! -d "$SCRATCH/hf_cache" ]; then
    mkdir -p "$SCRATCH/hf_cache"
fi

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# get Huggingface token from python
export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")

# training params
export TRAINING_SCRIPT
export CONFIG_PATH

# export params
export PYTHON_ENV
export INTERACTIVE

# debug nvidia
# force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_NCCL_USE_COMM_NONBLOCKING=1

# singolo nodo, 4 gpu, con e senza
export NCCL_IB_SL=1
export UCX_IB_SL=1
export NVSHMEM_IB_SL=1
export NVSHMEM_DISABLE_NCCL=1

# echo the params
echo "CURRENT_DIR: $CURRENT_DIR"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "PYTHON_ENV: $PYTHON_ENV"
echo "TRAINING_SCRIPT: $TRAINING_SCRIPT"

if [ "$INTERACTIVE" = "TRUE" ]; then
    echo "Running job interactively"
    bash ./helpers/train.slurm
else
    echo "NODES: $NODES"
    echo "GPU_PER_NODE: $GPU_PER_NODE"
    echo "LOGS_PATH: $LOGS_PATH"
    echo "MODULES: $MODULES"
    echo "TIME: $TIME"
    echo "ACCOUNT: $ACCOUNT"
    echo "PARTITION: $PARTITION"
    echo "JOB_NAME: $JOB_NAME"
    echo "STD_OUT: $STD_OUT"
    echo "STD_ERR: $STD_ERR"
    echo "EXCLUSIVE: $EXCLUSIVE"
    echo "CHAIN: $CHAIN"
    echo "Running job non-interactively"
    if [ $CHAIN -gt 1 ]; then
        # chain jobs by submitting the next job in the chain
        # capture the job id of the current job
        JOB_OUTPUT=$(sbatch -p $PARTITION \
            -A $ACCOUNT \
            --nodes=$NODES \
            --ntasks=$NODES \
            --time=$TIME \
            --job-name=$JOB_NAME \
            --output="$STD_OUT.0" \
            --error="$STD_ERR.0" \
            --ntasks-per-node=1 \
            --cpus-per-task=8 \
            --gres=gpu:$GPU_PER_NODE \
            $EXCLUSIVE \
            ./helpers/train.slurm)
        # extract the job id from "Submitted batch job 4751210"
        JOB_ID=$(echo $JOB_OUTPUT | grep "Submitted batch job" | awk '{print $4}')
        # chain the next job
        for i in $(seq 2 $CHAIN); do
            JOB_ID_AFTER=$(sbatch -p $PARTITION \
                -A $ACCOUNT \
                --nodes=$NODES \
                --ntasks=$NODES \
                --time=$TIME \
                --job-name=$JOB_NAME \
                --output="$STD_OUT.$i" \
                --error="$STD_ERR.$i" \
                --ntasks-per-node=1 \
                --cpus-per-task=8 \
                --gres=gpu:$GPU_PER_NODE \
                $EXCLUSIVE \
                --dependency=afterany:$JOB_ID \
                ./helpers/train.slurm | grep "Submitted batch job" | awk '{print $4}')
            echo "Chaining job $JOB_ID_AFTER after $JOB_ID"
            JOB_ID=$JOB_ID_AFTER
        done
    else
        sbatch -p $PARTITION \
            -A $ACCOUNT \
            --nodes=$NODES \
            --ntasks=$NODES \
            --time=$TIME \
            --job-name=$JOB_NAME \
            --output=$STD_OUT \
            --error=$STD_ERR \
            --ntasks-per-node=1 \
            --cpus-per-task=8 \
            --gres=gpu:$GPU_PER_NODE \
            $EXCLUSIVE \
            ./helpers/train.slurm
    fi
fi