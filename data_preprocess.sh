#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --ntasks-per-node=4
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --exclude=compute-0-[7,11,13]
#SBATCH -o /home/ajangid/HPLFlowNet/log_data_preprocess.txt
#SBATCH -e /home/ajangid/HPLFlowNet/err_data_preprocess.txt

set -x
set -u
set -e
python3 data_preprocess/process_kitti.py /home/ajangid/HPLFlowNet/raw_data/KITTI /home/ajangid/HPLFlowNet/dataset_processed/KITTI_processed_occ_final
