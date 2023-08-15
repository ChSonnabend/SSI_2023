#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=kill-shared
#SBATCH --time=1-00:00:00 ## time format is DD-HH:MM:SS

#SBATCH --cpus-per-task=6
#SBATCH --mem=120000 

#SBATCH --gres=gpu:NV-A30:1  ## request both GPUs in the GPU node.

#SBATCH --output=model_full_0p75.out

module purge
ml lang/Anaconda3 
source activate tf_gpu 
python full.py
