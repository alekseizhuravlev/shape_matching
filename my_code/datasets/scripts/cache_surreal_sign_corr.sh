#!/bin/bash

#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --array=0-19
#SBATCH --gres=gpu:1
#SBATCH --partition=mlgpu_medium
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/shape_matching/SLURM_logs/cache_surreal_%A_%a.out
#SBATCH --error=/home/s94zalek_hpc/shape_matching/SLURM_logs/cache_surreal_%A_%a.err

source /home/s94zalek_hpc/.bashrc
conda activate fmnet
cd /home/s94zalek_hpc/shape_matching
module load libGLU Xvfb
export PYTHONPATH=${PYTHONPATH}:/home/s94zalek_hpc/shape_matching

train_worker_count=$((SLURM_ARRAY_TASK_COUNT - 1))

num_evecs=32
net_name='signNet_remeshed_mass_6b_1ev_10_0.2_0.8'
dataset_name='SURREAL_augShapes_signNet_remeshed_mass_6b_1ev_10_0.2_0.8'

srun python my_code/datasets/cache_surreal_sign_corr.py  --num_evecs ${num_evecs} --n_workers ${SLURM_ARRAY_TASK_COUNT} --current_worker ${SLURM_ARRAY_TASK_ID} --net_path /home/s94zalek_hpc/shape_matching/my_code/experiments/sign_net/${net_name} --dataset_name ${dataset_name}

    