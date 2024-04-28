#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=/home/s94zalek/shape_matching/scripts/logs/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/home/s94zalek/shape_matching/scripts/error_logs/%j.err  # where to store error messages

# Exit on errors
set -o errexit

module load Anaconda3 libGLU Xvfb

# conda init bash
activate fmnet
cd /home/s94zalek/shape_matching
nvidia-smi

jupyter notebook --no-browser --port 5998 --ip $(hostname -f)
