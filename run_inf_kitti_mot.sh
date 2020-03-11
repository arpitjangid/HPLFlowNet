#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --ntasks-per-node=4
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --exclude=compute-0-[7,9,11,13]
#SBATCH -o /home/ajangid/HPLFlowNet/log_kitti_mot.txt
#SBATCH -e /home/ajangid/HPLFlowNet/error_kitti_mot.txt

set -x
set -u
set -e
time  \
	python3 main.py configs/test_ours_KITTI_MOT.yaml
