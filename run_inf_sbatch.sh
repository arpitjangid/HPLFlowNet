#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --exclude=compute-0-[7,11,13]
#SBATCH -o /home/ajangid/HPLFlowNet/log.txt
#SBATCH -e /home/ajangid/HPLFlowNet/error_log.txt

set -x
set -u
set -e
#module load singularity
#module load cuda-10.0
time  \
	#singularity exec --nv /containers/images/ubuntu-16.04-lts-GPU.img \
	python3 main.py configs/test_ours_KITTI.yaml
