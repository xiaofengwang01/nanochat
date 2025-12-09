#!/bin/bash


#SBATCH -J train              # The job name
#SBATCH -o slurm_log/%x-%j.out        # Write the standard output to file named 'ret-<job_number>.out'
#SBATCH -e slurm_log/%x-%j.err        # Write the standard error to file named 'ret-<job_number>.err'


#SBATCH -p r8nv-gpu-hw-80g               # Submit to 'r8nv-gpu-hw' Partitiion
#SBATCH -t 1-06:00:00                    # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#SBATCH --nodes=1                        # Request N nodes
#SBATCH --gres=gpu:8                     # Request M GPU per node
#SBATCH --gres-flags=enforce-binding     # CPU-GPU Affinity
#SBATCH --qos=gpu-normal                 # Request QOS Type

#- Log information

# echo current file path

echo "Current directory is $(pwd)"

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"
echo "$(df -h | grep -v tmpfs)"

#- Important settings!!!
# 1. Prevents RDMA resource exhaustion errors:
ulimit -l unlimited
# 2. Prevents virtual memory exhaustion errors, which are critical
#    when loading Large Language Models (LLMs):
ulimit -v unlimited
# 3. Increases the maximum number of open file descriptors to avoid
#    issues with too many concurrent connections or file accesses:
ulimit -n 65535
# 4. Raises the maximum number of user processes to support
#    large-scale parallel workloads:
ulimit -u 4125556

#- Load environments
source /tools/module_env.sh
module list                       # list modules loaded

##- Tools
module load cluster-tools/v1.0
module load slurm-tools/v1.0

##- language
# module load python3/3.8.16

##- CUDA
module load cuda-cudnn/12.4-9.1.1

##- virtualenv
# source xxxxx/activate

echo $(module list)              # list modules loaded
echo $(which gcc)
echo $(which python)
echo $(which python3)

# activate the uv virtual environment
source .venv/bin/activate
export WANDB_MODE=offline
export WANDB_RUN=nanochat

cluster-quota                    # nas quota

nvidia-smi --format=csv --query-gpu=name,driver_version,power.limit # gpu info

#- WARNING! DO NOT MODIFY your CUDA_VISIBLE_DEVICES
#- in `.bashrc`, `env.sh`, or your job script
echo "Using GPU(s) ${CUDA_VISIBLE_DEVICES}"                         # which GPUs
#- The CUDA_VISIBLE_DEVICES variable is assigned and specified by SLURM
echo "This job is assigned the following resources by SLURM:"
scontrol show jobid $SLURM_JOB_ID -dd | awk '/IDX/ {print $2, $4}'

##- Monitor
# The script continues executing other tasks while the following command will execute after a while
module load slurm-tools/v1.0
(sleep 3h && slurm-gpu-atop-log-stats $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES) &
echo "Main program continues to run. Monitoring information will be exported after three hours."

#- Main program execution
# set nanochat base dir
export NANOCHAT_BASE_DIR="/nfs_global/S/wangxiaofeng/.cache/nanochat"
# Number of processes/GPUs to use
NPROC_PER_NODE=8

# pretrain the d20 model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --run=$WANDB_RUN --save_every=2000
# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
# evaluate the model on CORE tasks
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

#- End
slurm-gpu-atop-log-stats $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
# This will overwrite any existing atop logs from previous runs.
# WARNING: If your program times out or is terminated by scancel,
#          the above script part might not execute correctly.
