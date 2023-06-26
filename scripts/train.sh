#!/bin/bash

# Pre-start
# checkmark font for fancy log
CHECK_MARK="\033[0;32m\xE2\x9C\x94\033[0m"
# usage text
USAGE="$(basename "$0") [-h --help] [-l --language-model LANG_MODEL_NAME] [-d --debug] [-p --precision PRECISION]
[-c --cpu] [-g --devices DEVICES] [-n --nodes NODES] [-m --gpu-mem GPU_MEM] [-s --strategy STRATEGY]
[-o --offline] [-t --test] [--config-path CONFIG_PATH] [--checkpoint CHECKPOINT_PATH] [-w --wandb WANDB_PROJECT] OVERRIDES

where:
    -h --help             Show this help text
    -l --language-model   Language model name (one of the models from HuggingFace)
    -d --debug            Run in debug mode (no GPU and wandb offline)
    -p --precision        Training precision, default 16.
    -c --cpu              Use CPU instead of GPU.
    -g --devices          How many GPU to use, default 1. If 0, use CPU.
    -n --nodes            How many nodes to use, default 1.
    -m --gpu-mem          Minimum GPU memory required in MB (default: 8000). If less that this,
                          training will wait until there is enough space.
    -s --strategy         Strategy to use for distributed training, default NULL.
    -o --offline          Run the experiment offline
    -v --print            Print the config
    -t --test             Run only the test phase
    -w --wandb            The wandb project name
    --config-path         Run a specific config file
    --checkpoint          Run a specific checkpoint
    OVERRIDES             Overrides for the experiment, in the form of key=value.
                          For example, 'model_name=bert-base-uncased'.
Example:
  bash scripts/train.sh --config-path conf/finetune_iterable_in_batch.yaml -l intfloat/e5-base-v2
"

# Transform long options to short ones
for arg in "$@"; do
  shift
  case "$arg" in
  '--help') set -- "$@" '-h' ;;
  '--language-model') set -- "$@" '-l' ;;
  '--debug') set -- "$@" '-d' ;;
  '--precision') set -- "$@" '-p' ;;
  '--cpu') set -- "$@" '-c' ;;
  '--devices') set -- "$@" '-g' ;;
  '--nodes') set -- "$@" '-n' ;;
  '--gpu-mem') set -- "$@" '-m' ;;
  '--strategy') set -- "$@" '-s' ;;
  '--offline') set -- "$@" '-o' ;;
  '--print') set -- "$@" '-v' ;;
  '--test') set -- "$@" '-t' ;;
  '--config-path') set -- "$@" '-a' ;;
  '--checkpoint') set -- "$@" '-k' ;;
  '--wandb') set -- "$@" '-w' ;;
  *) set -- "$@" "$arg" ;;
  esac
done

# check for named params
#while [ $OPTIND -le "$#" ]; do
while getopts ":hl:dp:cg:n:m:s:ovta:k:w:" opt; do
  case $opt in
  h)
    printf "%s$USAGE" && exit 0
    ;;
  l)
    LANG_MODEL_NAME="$OPTARG"
    ;;
  d)
    DEV_RUN="True"
    ;;
  p)
    PRECISION="$OPTARG"
    ;;
  c)
    USE_CPU="True"
    ;;
  g)
    DEVICES="$OPTARG"
    ;;
  n)
    NODES="$OPTARG"
    ;;
  m)
    GPU_MEM="$OPTARG"
    ;;
  s)
    STRATEGY="$OPTARG"
    ;;
  o)
    WANDB="offline"
    ;;
  v)
    PRINT_CONFIG="++print_config=True"
    ;;
  t)
    ONLY_TEST="True"
    ;;
  a)
    CONFIG_PATH="$OPTARG"
    ;;
  k)
    CHECKPOINT_PATH="$OPTARG"
    ;;
  w)
    WANDB_PROJECT="$OPTARG"
    ;;
  \?)
    echo "Invalid option -$OPTARG" >&2 && echo "$USAGE" && exit 0
    ;;
  esac
done

# shift for overrides
shift $((OPTIND - 1))
# split overrides into key=value pairs
OVERRIDES=$(echo "$@" | sed -e 's/ /\n/g')

if [ "$PRINT_CONFIG" ]; then
  OVERRIDES="$OVERRIDES $PRINT_CONFIG"
fi

if [ "$LANG_MODEL_NAME" ]; then
  OVERRIDES="$OVERRIDES model.language_model=$LANG_MODEL_NAME"
fi

# PRELIMINARIES
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/bin/activate golden

# Default device is GPU
ACCELERATOR="gpu"

# if -d is not specified, DEV_RUN is False
if [ -z "$DEV_RUN" ]; then
  # default value
  DEV_RUN=False
fi

# if -p is not specified, PRECISION is 16
if [ -z "$PRECISION" ]; then
  # default value
  PRECISION=16
fi

# if -c is not specified, USE_CPU is False
if [ -z "$USE_CPU" ]; then
  # default value
  USE_CPU=False
fi

# if -g is not specified, DEVICES is 1
if [ -z "$DEVICES" ]; then
  # default value
  DEVICES=1
fi

# if -n is not specified, NODES is 1
if [ -z "$NODES" ]; then
  # default value
  NODES=1
fi

# if -m is not specified, GPU_MEM is not limited
if [ -z "$GPU_MEM" ]; then
  # default value
  GPU_MEM=0
fi

# if -o is not specified, WANDB is "online"
if [ -z "$WANDB" ]; then
  # default value
  WANDB="online"
fi

# if -a is not specified, CONFIG_PATH is "null"
if [ -z "$CONFIG_PATH" ]; then
  CONFIG_PATH=""
else
  # split the last part of the path
  CONFIG_NAME=$(echo "$CONFIG_PATH" | rev | cut -d'/' -f1 | rev)
  # remove the extension
  CONFIG_NAME=$(echo "$CONFIG_NAME" | cut -d'.' -f1)
  # get the path
  CONFIG_PATH=$(echo "$CONFIG_PATH" | rev | cut -d'/' -f2- | rev)
  # add the absolute path to the actual project directory to the config path
  CONFIG_PATH=$(realpath "$CONFIG_PATH")
  CONFIG_PATH="--config-path $CONFIG_PATH --config-name $CONFIG_NAME"
fi

# if -k is not specified, CHECKPOINT_PATH is "null"
if [ -z "$CHECKPOINT_PATH" ]; then
  CHECKPOINT_PATH=""
else
  OVERRIDES="$OVERRIDES +evaluation.checkpoint_path=$CHECKPOINT_PATH"
fi

if [ -z "$WANDB_PROJECT" ]; then
  WANDB_PROJECT=""
else
  OVERRIDES="$OVERRIDES project_name=$WANDB_PROJECT"
fi

# if -t is not specified, ONLY_TEST is False
if [ -z "$ONLY_TEST" ]; then
  ONLY_TEST="False"
else
  OVERRIDES="$OVERRIDES ++train.only_test=True"
  WANDB="offline"
fi

# CHECK FOR BOOLEAN PARAMS
# if -d then GPU is not required and no output dir
if [ "$DEV_RUN" = "True" ]; then
  WANDB="offline"
  DEVICES=1
  PRECISION=32
  ACCELERATOR="cpu"
  NODES=1
  USE_CPU="True"
fi

# if -s is not specified, STRATEGY is None
if [ -z "$STRATEGY" ]; then
  # default value
  STRATEGY="auto"
fi

# if -g DEVICES is 0 (no GPU) and PRECISION is 32
if [ "$USE_CPU" = "True" ]; then
  # default value
  DEVICES=1
  PRECISION=32
  ACCELERATOR="cpu"
  STRATEGY="auto"
fi

if type nvidia-smi >/dev/null 2>&1; then
  FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0 | grep -Eo "[0-9]+")
else
  FREE_MEM=0
  GPU_MEM=0
  PRECISION=32
  ACCELERATOR="cpu"
  if [ $USE_CPU = "False" ]; then
    echo -e "GPU not found, fallback to CPU.\n"
  fi
  USE_CPU="True"
fi
GPU_RAM_MESSAGE=""

# echo "$OVERRIDES" | fold -w 70 -s

cat <<EOF

.-------------------------------------------------------------------------.
|                                                                         | 
|         ██████   ██████  ██      ██████  ███████ ███    ██              |
|        ██       ██    ██ ██      ██   ██ ██      ████   ██              |
|        ██   ███ ██    ██ ██      ██   ██ █████   ██ ██  ██              |      
|        ██    ██ ██    ██ ██      ██   ██ ██      ██  ██ ██              |
|         ██████   ██████  ███████ ██████  ███████ ██   ████              |                                       
|                                                                         |  
|                                                                         |
|  ██████  ███████ ████████ ██████  ██ ███████ ██    ██ ███████ ██████    |
|  ██   ██ ██         ██    ██   ██ ██ ██      ██    ██ ██      ██   ██   |
|  ██████  █████      ██    ██████  ██ █████   ██    ██ █████   ██████    |
|  ██   ██ ██         ██    ██   ██ ██ ██       ██  ██  ██      ██   ██   |
|  ██   ██ ███████    ██    ██   ██ ██ ███████   ████   ███████ ██   ██   |
|                                                                         |       
'-------------------------------------------------------------------------'    

             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣴⣶⣾⠿⠶⢶⣶⣤⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                  
             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣴⡿⠟⠋⠉⠀⠀⠀⠀⠀⠀⠈⠉⠉⠻⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀                  
             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⠟⠉⠀⠀⣠⣤⣶⣷⢀⡀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀                  
             ⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⡟⠁⠀⠀⠀⠘⠛⠁⠀⣀⣼⣿⣦⠀⠀⠀⠀⠀⠙⣦⣮⣹⡆⠀⠀⠀⠀⠀⠀                  
             ⠀⠀⠀⠀⠀⠀⠀⣠⣾⡿⠀⠀⠀⠀⠀⠀⠀⠰⢿⡿⠿⠟⠛⠉⠀⠀⠀⠀⠀⠈⠙⠛⠳⠶⣤⣄⡀⠀⠀                  
             ⠀⠀⠀⠀⠀⢀⣾⡿⠛⢀⡀⠀⠀⢀⣠⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠚⣉⣉⣉⣙⣧⠀                  
             ⠀⠀⠀⠀⠀⣼⠏⠀⠀⠀⣷⠀⣴⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⣠⠀⠀⠀⣿⠀⢿⣿⣿⣿⣿⣿⠀                  
             ⠀⠀⠀⠀⢀⡿⠀⠀⠀⠀⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣯⣤⡾⠃⠀⠀⠀⠈⠻⠶⣾⣿⣿⣿⣿⠄                  
             ⠀⠀⠀⠀⣸⡇⠀⠀⠀⢰⣿⣿⠀⡄⠀⠀⠀⠀⠀⠀⠀⠀⠘⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⡿⠀                  
             ⠀⠀⠀⣰⡿⠀⠀⠀⠀⠈⢹⣿⡼⠁⠀⠀⠀⠀⠀⢠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣾⣿⣿⠟⠁⠀                  
             ⠀⠀⣰⡿⢡⠂⠀⠀⠀⠀⠀⣿⣇⣰⣂⣠⠀⠀⠀⣟⣀⣀⠀⠀⠀⠀⠀⣠⣤⠶⠿⠿⠿⠛⠿⣧⠀⠀⠀                  
             ⣰⡿⢠⣏⡔⠀⠀⠀⠀⠀⢿⣿⣿⣿⡇⢠⡀⠀⢿⣿⣿⣿⣍⣙⠛⢻⣏⣤⠾⣦⡀⠀⠀⠀⢹⡇⠀⠀⠀                  
             ⣿⣷⣿⣿⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⣿⣿⣧⠀⠈⠻⢿⣿⣿⣿⣿⣿⣿⡿⣶⣾⣿⣶⣦⣴⡟⠁⠀⠀⠀                  
             ⠹⢿⣿⣿⡇⠀⠀⠀⠀⠀⠀⢹⣿⣿⣿⣿⡏⠀⠀⠀⠀⠈⠉⠉⠉⠙⠿⠟⠁⠈⠙⠛⢻⡿⠀⠀⠀⠀⠀                  
             ⠀⠀⠀⠙⠻⣧⡀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⠋⠀⠀⡴⣾⣿⣷⣶⣤⣤⣤⣤⣤⣶⠤⠶⠛⠀⠀⠀⠀⠀                  
             ⠀⠀⠀⠀⠀⠙⢿⣶⣤⣤⣤⣴⣿⣿⣿⣿⠃⠀⠀⠀⢰⡟⢿⣿⣿⡿⠻⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀                  
             ⠀⠀⠀⠀⠀⠀⠀⠉⠙⠛⠛⠋⠁⠙⢿⣿⠀⢠⠄⠀⡼⠀⢸⣿⢹⡇⠀⢹⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀                  
             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣶⡟⣸⠁⠃⠀⣿⣿⡀⠧⠀⣸⣿⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀                  
             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣟⣴⣿⠀⠀⠀⣿⠹⡇⠀⢠⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                  
             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠛⣻⡿⢿⣆⢠⣿⠀⠁⢀⣾⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                  
             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠾⠋⠁⢈⣿⣿⣿⡇⠀⣾⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                  
             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⡿⠋⠉⢿⣾⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                  
             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠋⠀⠀⠀⠈⠿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                  

                Golden Retriever Training Script
EOF


# WAITING FOR VRAM STUFF
chars="/-\|"
if [ "$FREE_MEM" -lt $GPU_MEM ]; then
  # echo -n "$GPU_RAM_MESSAGE"
  GPU_RAM_MESSAGE="\\rWaiting for at least $GPU_MEM MB of VRAM "
  echo -ne "$GPU_RAM_MESSAGE"
fi
while [ "$FREE_MEM" -lt $GPU_MEM ]; do
  FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0 | grep -Eo "[0-9]+")
  for ((i = 0; i < ${#chars}; i++)); do
    sleep 1
    echo -ne "${chars:$i:1} $GPU_RAM_MESSAGE"
  done
done

# if DEV_RUN then GPU is not required
if [ "$DEV_RUN" = "True" ]; then
  echo -n "Debug run started, ignoring GPU memory. "
  GPU_RAM_MESSAGE=""
fi

echo -e "$GPU_RAM_MESSAGE${CHECK_MARK} Starting.\n"

# if you use the `GenerativeDataset` class
# you may want to set `TOKENIZERS_PARALLELISM` to `false`
export TOKENIZERS_PARALLELISM=false

DIRPATH=$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE:-$0}")")")/src
export PYTHONPATH="$DIRPATH"

export HYDRA_FULL_ERROR=1

if [ "$DEV_RUN" = "True" ]; then
  python goldenretriever/trainer/train.py \
    $CONFIG_PATH \
    "train.pl_trainer.fast_dev_run=$DEV_RUN" \
    "train.pl_trainer.devices=$DEVICES" \
    "train.pl_trainer.accelerator=$ACCELERATOR" \
    "train.pl_trainer.num_nodes=$NODES" \
    "train.pl_trainer.strategy=$STRATEGY" \
    "train.pl_trainer.precision=$PRECISION" \
    "hydra.run.dir=." \
    "hydra.output_subdir=null" \
    "hydra/job_logging=disabled" \
    "hydra/hydra_logging=disabled" \
    $OVERRIDES
else
  python goldenretriever/trainer/train.py \
    $CONFIG_PATH \
    "train.pl_trainer.fast_dev_run=$DEV_RUN" \
    "train.pl_trainer.devices=$DEVICES" \
    "train.pl_trainer.accelerator=$ACCELERATOR" \
    "train.pl_trainer.num_nodes=$NODES" \
    "train.pl_trainer.strategy=$STRATEGY" \
    "train.pl_trainer.precision=$PRECISION" \
    "logging.wandb_arg.mode=$WANDB" \
    $OVERRIDES
fi
