#!/bin/bash

#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --partition=mlgpu_medium
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/shape_matching/SLURM_logs/test_diffusion_%A_%a.out
#SBATCH --error=/home/s94zalek_hpc/shape_matching/SLURM_logs/test_diffusion_%A_%a.err

source /home/s94zalek_hpc/.bashrc
conda activate fmnet
cd /home/s94zalek_hpc/shape_matching
module load libGLU Xvfb

# run this command export PYTHONPATH="${PYTHONPATH}:/home/s94zalek_hpc/shape_matching"
# before running this script
export PYTHONPATH=${PYTHONPATH}:/home/s94zalek_hpc/shape_matching


experiment_name='pair_augShapes_signNet_remeshed_4b_mass_10_0.2_0.8'
checkpoint_name='checkpoint_99.pt'

# put all dataset names and splits in a list
job_list=(
    'FAUST_orig test'
    'FAUST_orig_pair test'
    'FAUST_r test'
    'FAUST_r_pair test'
    'FAUST_a test'
    'FAUST_a_pair test'
    'SHREC19_r_pair test'
    'SCAPE_r_pair test'
    'SCAPE_a_pair test'
    'FAUST_r train'
    'FAUST_orig train'
    'SURREAL train'
)

# worker id = id of the current job in the job list
worker_id=$SLURM_ARRAY_TASK_ID

# get the current job from the job list
current_job=${job_list[$worker_id]}
# split the job into dataset name and split
dataset_name=$(echo $current_job | cut -d ' ' -f 1)
split=$(echo $current_job | cut -d ' ' -f 2)

echo "Running job $worker_id: dataset_name=$dataset_name, split=$split"

# run the job
srun python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name $dataset_name --split $split



