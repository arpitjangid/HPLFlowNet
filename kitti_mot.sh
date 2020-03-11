#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --exclude=compute-0-[7,9,11]
#SBATCH -o /home/ajangid/HPLFlowNet/log_mot.txt
#SBATCH -e /home/ajangid/HPLFlowNet/error_mot.txt

set -x
set -u
set -e
#module load singularity
#module load cuda-10.0
time  \
	
	python kitti_mot_sample.py
