#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --ntasks-per-node=4
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --exclude=compute-0-[7,11,13]
#SBATCH -o /home/ajangid/HPLFlowNet/log_nuscenes_diff10.txt
#SBATCH -e /home/ajangid/HPLFlowNet/error_nuscenes_diff10.txt

set -x
set -u
set -e
time  \
	python3 main.py configs/test_ours_nuScenes.yaml
