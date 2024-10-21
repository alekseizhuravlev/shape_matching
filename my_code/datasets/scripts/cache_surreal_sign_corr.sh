#!/bin/bash

#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --array=0-99
#SBATCH --mem=50G
#SBATCH --partition=intelsr_medium
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/shape_matching/SLURM_logs/cache_surreal_%A_%a.out
#SBATCH --error=/home/s94zalek_hpc/shape_matching/SLURM_logs/cache_surreal_%A_%a.err

source /home/s94zalek_hpc/.bashrc
conda activate pyshot_new
cd /home/s94zalek_hpc/shape_matching
module load libGLU Xvfb
export PYTHONPATH=${PYTHONPATH}:/home/s94zalek_hpc/shape_matching

train_worker_count=$((SLURM_ARRAY_TASK_COUNT - 1))

num_evecs=128
net_name='signNet_128_remeshed_mass_6b_1-2-2-2ev_10_0.2_0.8'
dataset_name='SURREAL_128_1-2-2-2ev_template_remeshed_augShapes_bbox'

regularization_lambda=-1
partial=-1

# num_evecs=32
# net_name='test_partial_isoRemesh_shot'
# dataset_name='partial_isoRemesh_shot_lambda_0.01_anisRemesh_holes_partial_0.8'
# regularization_lambda=0.01
# partial=0.8

template_type='remeshed'
pair_type='template'
n_pairs=1
centering='bbox'


srun python my_code/datasets/cache_surreal_sign_corr.py  --num_evecs ${num_evecs} --n_workers ${SLURM_ARRAY_TASK_COUNT} --current_worker ${SLURM_ARRAY_TASK_ID} --net_path /home/s94zalek_hpc/shape_matching/my_code/experiments/sign_net/${net_name} --dataset_name ${dataset_name} --template_type ${template_type} --pair_type ${pair_type} --n_pairs ${n_pairs} --regularization_lambda ${regularization_lambda} --partial ${partial} --centering ${centering}


# rewrite as line of code with values instead of variables
# python my_code/datasets/cache_surreal_sign_corr.py  --num_evecs 32 --n_workers 1 --current_worker 0 --net_path /home/s94zalek_hpc/shape_matching/my_code/experiments/sign_net/test_partial_anisRemesh_shot --dataset_name test_partial_anisRemesh_shot_lambda_0.01_anisRemesh_holes_partial_0.8 --template_type remeshed --pair_type template --n_pairs 1 --regularization_lambda 0.01 --partial 0.8  