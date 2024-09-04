#!/bin/bash

#SBATCH -n 1
#SBATCH -t 7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mlgpu_long
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/shape_matching/SLURM_logs/train_diffusion_%j.out
#SBATCH --error=/home/s94zalek_hpc/shape_matching/SLURM_logs/train_diffusion_%j.err

source /home/s94zalek_hpc/.bashrc
conda activate fmnet
cd /home/s94zalek_hpc/shape_matching
module load libGLU Xvfb

# run this command export PYTHONPATH="${PYTHONPATH}:/home/s94zalek_hpc/shape_matching"
# before running this script
export PYTHONPATH=${PYTHONPATH}:/home/s94zalek_hpc/shape_matching


srun python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/train_pipeline.py



