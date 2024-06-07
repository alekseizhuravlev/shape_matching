#!/bin/bash
#SBATCH --partition=mlgpu_medium
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/shape_matching/SLURM_logs/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/home/s94zalek_hpc/shape_matching/SLURM_logs/%j.err  # where to store error messages

# Exit on errors
set -o errexit

source /home/s94zalek_hpc/.bashrc

conda activate fmnet
cd /home/s94zalek_hpc/shape_matching

# load module
module load libGLU Xvfb

nvidia-smi

jupyter notebook --no-browser --port 5998 --ip $(hostname -f)

