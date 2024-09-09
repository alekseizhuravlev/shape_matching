#!/bin/bash

#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --array=0-20
#SBATCH --gres=gpu:1
#SBATCH --partition=mlgpu_medium
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/shape_matching/SLURM_logs/generate_pair_fmaps_%A_%a.out
#SBATCH --error=/home/s94zalek_hpc/shape_matching/SLURM_logs/generate_pair_fmaps_%A_%a.err

source /home/s94zalek_hpc/.bashrc
conda activate fmnet
cd /home/s94zalek_hpc/shape_matching
module load libGLU Xvfb
export PYTHONPATH=${PYTHONPATH}:/home/s94zalek_hpc/shape_matching


data_dir_in='/lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL/train/SURREAL_evecs_10_augShapes_signNet_remeshed_mass_6b_1ev_10_0.2_0.8/train'
data_dir_out='/lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL_pair/'
dataset_name='pair_5_augShapes_signNet_remeshed_mass_6b_1ev_10_0.2_0.8'
n_pairs=5

srun python my_code/datasets/generate_pair_fmaps.py --curr_worker ${SLURM_ARRAY_TASK_ID} --data_dir_in ${data_dir_in} --data_dir_out ${data_dir_out} --dataset_name ${dataset_name} --n_pairs ${n_pairs}

    