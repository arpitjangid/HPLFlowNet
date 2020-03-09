#!/usr/bin/env bash
#SBATCH --nodes=1
# SBATCH --partition=GPU
# SBATCH --ntasks-per-node=4
# SBATCH --time=10:00:00
# SBATCH --gres=gpu:1
# SBATCH --mem=16G
# SBATCH --exclude=compute-0-[7,11,13]
#SBATCH -o /home/ajangid/HPLFlowNet/log_test.txt
#SBATCH -e /home/ajangid/HPLFlowNet/err_test.txt

set -x
set -u
set -e

module load cuda-10.0
python3 main.py configs/test_ours_nuScenes.yaml