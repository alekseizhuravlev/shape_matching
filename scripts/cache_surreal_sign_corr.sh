#!/bin/bash

#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --array=0-19
#SBATCH --mem=50G
#SBATCH --partition=intelsr_medium
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/shape_matching/SLURM_logs/cache_surreal_%A_%a.out
#SBATCH --error=/home/s94zalek_hpc/shape_matching/SLURM_logs/cache_surreal_%A_%a.err

source /home/s94zalek_hpc/.bashrc
conda activate fmnet
cd /home/s94zalek_hpc/shape_matching
module load libGLU Xvfb

train_worker_count=$((SLURM_ARRAY_TASK_COUNT - 1))

num_evecs=32

srun python my_code/datasets/cache_surreal_sign_corr.py  --num_evecs ${num_evecs} --n_workers ${SLURM_ARRAY_TASK_COUNT} --current_worker ${SLURM_ARRAY_TASK_ID} --net_path /home/s94zalek_hpc/shape_matching/my_code/experiments/sign_net/signNet_remeshed_10_0.5_1 --dataset_name SURREAL_augShapes_signNet_remeshed_10_0.5_1
# srun python my_code/datasets/cache_surreal_sign_corr.py  --num_evecs ${num_evecs} --n_workers ${SLURM_ARRAY_TASK_COUNT} --current_worker ${SLURM_ARRAY_TASK_ID} --net_path /home/s94zalek_hpc/shape_matching/my_code/experiments/sign_net/signNet_remeshed_4b_mass_10_0.5_1 --dataset_name SURREAL_augShapes_mass_signNet_remeshed_10_0.5_1

    