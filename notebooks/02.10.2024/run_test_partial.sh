#!/bin/bash

#SBATCH -n 1
#SBATCH -t 1-00:00:00
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --partition=mlgpu_medium
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/shape_matching/SLURM_logs/test_partial_on_train_%A_%a.out
#SBATCH --error=/home/s94zalek_hpc/shape_matching/SLURM_logs/test_partial_on_train_%A_%a.err

source /home/s94zalek_hpc/.bashrc
conda activate fmnet
cd /home/s94zalek_hpc/shape_matching
module load libGLU Xvfb

# run this command export PYTHONPATH="${PYTHONPATH}:/home/s94zalek_hpc/shape_matching"
# before running this script
export PYTHONPATH=${PYTHONPATH}:/home/s94zalek_hpc/shape_matching


# put all dataset names and splits in a list
job_list=(
    "partial_0.8_5k_32_1_lambda_0.1_xy"
    "partial_0.8_5k_32_1_lambda_0.01_xy"
    "partial_0.8_5k_32_1_lambda_0.1_yx"
    "partial_0.8_5k_32_1_lambda_0.01_yx"

    "partial_0.8_5k_32_2_lambda_0.1_xy"
    "partial_0.8_5k_32_2_lambda_0.01_xy"
    "partial_0.8_5k_32_2_lambda_0.1_yx"
    "partial_0.8_5k_32_2_lambda_0.01_yx"
)

# worker id = id of the current job in the job list
worker_id=$SLURM_ARRAY_TASK_ID

# get the current job from the job list
experiment_name=${job_list[$worker_id]}

dataset_name='training_data'
split='train'

checkpoint_name='epoch_99'
num_iters_avg=64
num_samples_median=10
confidence_threshold=0.2
num_iters_dataset=150

log_subdir="test_partial_on_train_data"


echo "Testing experiment $experiment_name with checkpoint $checkpoint_name"
echo "Running job $worker_id: dataset_name=$dataset_name, split=$split"
echo "Log directory: $log_subdir"
# run the job

python /home/s94zalek_hpc/shape_matching/notebooks/02.10.2024/test_partial_on_train_data.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name $dataset_name --split $split --num_iters_avg $num_iters_avg --num_samples_median $num_samples_median --confidence_threshold $confidence_threshold --num_iters_dataset $num_iters_dataset --log_subdir $log_subdir

