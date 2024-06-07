#!/bin/bash

#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --array=0-14
#SBATCH --partition=vlm_medium
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/shape_matching/SLURM_logs/cache_surreal_%A_%a.out
#SBATCH --error=/home/s94zalek_hpc/shape_matching/SLURM_logs/cache_surreal_%A_%a.err

source /home/s94zalek_hpc/.bashrc
conda activate fmnet
cd /home/s94zalek_hpc/shape_matching
module load libGLU Xvfb

train_worker_count=$((SLURM_ARRAY_TASK_COUNT - 1))

n_body_types_male=50000
n_body_types_female=50000
n_poses_straight=85000
n_poses_bent=15000
num_evecs=32

if [ ${SLURM_ARRAY_TASK_ID} -eq ${train_worker_count} ]; then
    srun python my_code/datasets/cache_surreal.py --save_test_set --n_workers ${SLURM_ARRAY_TASK_COUNT} --current_worker ${SLURM_ARRAY_TASK_ID} --n_body_types_male ${n_body_types_male} --n_body_types_female ${n_body_types_female} --n_poses_straight ${n_poses_straight} --n_poses_bent ${n_poses_bent} --num_evecs ${num_evecs}
else
    srun python my_code/datasets/cache_surreal.py --n_workers ${train_worker_count} --current_worker ${SLURM_ARRAY_TASK_ID} --n_body_types_male ${n_body_types_male} --n_body_types_female ${n_body_types_female} --n_poses_straight ${n_poses_straight} --n_poses_bent ${n_poses_bent} --num_evecs ${num_evecs}
fi
    