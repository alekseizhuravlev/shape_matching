#!/bin/bash

#SBATCH -n 1
#SBATCH -t 14:00:00
#SBATCH --array=0-5
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


experiment_name='single_64_2-4ev_64-128-128_remeshed'
num_iters_avg=32
num_samples_median=8
confidence_threshold=0.1


# checkpoint_name='checkpoint_95.pt'
checkpoint_name='epoch_99'

# put all dataset names and splits in a list
job_list=(
    # 'FAUST_orig test'
    # 'FAUST_r test'
    # 'FAUST_a test'
    # 'FAUST_r train'
    # 'FAUST_orig train'
    # 'SURREAL train'

    'FAUST_orig_pair test'
    'FAUST_r_pair test'
    'FAUST_a_pair test'
    'SHREC19_r_pair test'
    'SCAPE_r_pair test'
    'SCAPE_a_pair test'

    # 'DT4D_inter_pair test'
    # 'DT4D_intra_pair test'
)

# worker id = id of the current job in the job list
worker_id=$SLURM_ARRAY_TASK_ID

# get the current job from the job list
current_job=${job_list[$worker_id]}
# split the job into dataset name and split
dataset_name=$(echo $current_job | cut -d ' ' -f 1)
split=$(echo $current_job | cut -d ' ' -f 2)

echo "Testing experiment $experiment_name with checkpoint $checkpoint_name"
echo "Running job $worker_id: dataset_name=$dataset_name, split=$split"

# run the job


# with template

# no smoothing
srun python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_pair_template.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name $dataset_name --split $split --num_iters_avg $num_iters_avg --num_samples_median $num_samples_median --confidence_threshold $confidence_threshold

# taubin 5
srun python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_pair_template_smooth.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name $dataset_name --split $split --smoothing_type taubin --smoothing_iter 5 --num_iters_avg $num_iters_avg --num_samples_median $num_samples_median --confidence_threshold $confidence_threshold

# laplacian 3
# srun python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_pair_template_smooth.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name $dataset_name --split $split --smoothing_type laplacian --smoothing_iter 3


# pairwise

# no smoothing
# srun python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name $dataset_name --split $split --num_iters_avg $num_iters_avg

# write the line above ,with values instead of variables
# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_pair_template.py --experiment_name single_48_remeshed_noAcc_yx_64_128_128 --checkpoint_name checkpoint_90.pt --dataset_name SHREC19_r_pair --split test --num_iters_avg 32 --num_samples_median 4

# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond_smooth.py --experiment_name pair_5_xy_distributed --checkpoint_name epoch_99 --dataset_name SHREC19_r_pair --split test --num_iters_avg 16 --smoothing_type taubin --smoothing_iter 5

# taubin 5
# srun python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond_smooth.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name $dataset_name --split $split --smoothing_type taubin --smoothing_iter 5 --num_iters_avg $num_iters_avg


