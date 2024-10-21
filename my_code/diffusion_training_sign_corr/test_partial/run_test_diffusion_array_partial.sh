#!/bin/bash

#SBATCH -n 1
#SBATCH -t 1-00:00:00
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --partition=mlgpu_medium
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/shape_matching/SLURM_logs/test_diffusion_partial_%A_%a.out
#SBATCH --error=/home/s94zalek_hpc/shape_matching/SLURM_logs/test_diffusion_partial_%A_%a.err

source /home/s94zalek_hpc/.bashrc
conda activate pyshot_new
cd /home/s94zalek_hpc/shape_matching
module load libGLU Xvfb

# run this command export PYTHONPATH="${PYTHONPATH}:/home/s94zalek_hpc/shape_matching"
# before running this script
export PYTHONPATH=${PYTHONPATH}:/home/s94zalek_hpc/shape_matching



experiment_name=$1
checkpoint_name=$2

num_iters_avg=64
num_samples_median=10
confidence_threshold=0.2

log_subdir="logs_partial_symm"



# put all dataset names and splits in a list
job_list=(
    'SHREC16_cuts_pair_noSingle test'
    'SHREC16_holes_pair_noSingle test'
)

# worker id = id of the current job in the job list
worker_id=$SLURM_ARRAY_TASK_ID

current_job=${job_list[$worker_id]}
dataset_name=$(echo $current_job | cut -d ' ' -f 1)
split=$(echo $current_job | cut -d ' ' -f 2)


echo "Testing experiment $experiment_name with checkpoint $checkpoint_name"
echo "Running job $worker_id: dataset_name=$dataset_name, split=$split"
echo "Log directory: $log_subdir"


# no smoothing
srun python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test_partial/test_partial.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name $dataset_name --split $split --num_iters_avg $num_iters_avg --num_samples_median $num_samples_median --confidence_threshold $confidence_threshold --log_subdir $log_subdir 

# rewrite the line above with values instead of variables
# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test_partial/test_partial.py --experiment_name partial_anisRemesh_shot_lambda_0.01_anisRemesh_holes_partial_0.8_xy --checkpoint_name epoch_99 --dataset_name SHREC16_holes_pair_noSingle --split test --num_iters_avg 32 --num_samples_median 4 --confidence_threshold 0.2 --log_subdir logs_test --reduced


