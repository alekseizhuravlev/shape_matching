#!/bin/bash

#SBATCH -n 1
#SBATCH -t 7-00:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:2
#SBATCH --partition=mlgpu_long
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/shape_matching/SLURM_logs/train_diffusion_%j.out
#SBATCH --error=/home/s94zalek_hpc/shape_matching/SLURM_logs/train_diffusion_%j.err

source /home/s94zalek_hpc/.bashrc
conda activate fmnet
cd /home/s94zalek_hpc/shape_matching
module load libGLU Xvfb
export PYTHONPATH=${PYTHONPATH}:/home/s94zalek_hpc/shape_matching


# experiment_name='single_64_1-1ev_64-128-128_remeshed'
# dataset_name='SURREAL_64_template_remeshed_augShapes'


experiment_name='test_runtime_96_SMAL'
dataset_name='SMAL_nocat_96_SMAL_isoRemesh_0.2_0.8_nocat_1-2-4ev'


fmap_direction='yx'
sample_size=96

block_out_channels='64,128,128'


# experiment_name='single_32_SMAL_nocat_32_SMAL_isoRemesh_0.2_0.8_nocat_2ev_aug'
# dataset_name='SMAL_nocat_32_SMAL_isoRemesh_0.2_0.8_nocat_2ev_aug'

# fmap_direction='yx'
# sample_size=32

# block_out_channels='32,64,64'


# experiment_name='partial_isoRemesh_shot_lambda_0.01_anisRemesh_holes_partial_0.8_yx'
# dataset_name='partial_isoRemesh_shot_lambda_0.01_anisRemesh_holes_partial_0.8'

# fmap_direction='yx'
# sample_size=32

# block_out_channels='32,64,64'

down_block_types='DownBlock2D,AttnDownBlock2D,AttnDownBlock2D'
up_block_types='AttnUpBlock2D,AttnUpBlock2D,UpBlock2D'


# cp -r /lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL/train/${dataset_name} /tmp

# make directory ${dataset_name}/train in /tmp
mkdir -p /tmp/${dataset_name}/train

echo "Copying data to /tmp/${dataset_name}/train"

cp -r /lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL/train/${dataset_name}/train/C_gt_yx.pt /tmp/${dataset_name}/train
cp -r /lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL/train/${dataset_name}/train/evecs_cond_first.pt /tmp/${dataset_name}/train
cp -r /lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL/train/${dataset_name}/train/evecs_cond_second.pt /tmp/${dataset_name}/train
cp -r /lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL/train/${dataset_name}/config.yaml /tmp/${dataset_name}

echo "Data copied to /tmp/${dataset_name}/train"

# sample a random integer between 0 and 1000
port=$RANDOM

# --mixed_precision fp16 

srun accelerate launch --main_process_port ${port} /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/train_pipeline_distributed.py --experiment_name ${experiment_name} --dataset_name ${dataset_name} --fmap_direction ${fmap_direction} --sample_size ${sample_size} --block_out_channels ${block_out_channels} --down_block_types ${down_block_types} --up_block_types ${up_block_types} 

# rewrite the code above to include all arguments in the command line

# cp -r /lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL/train/SURREAL_64_1-2ev_template_remeshed_augShapes /tmp
# accelerate launch /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/train_pipeline_distributed.py --experiment_name single_64_1-2ev_64-128-128_remeshed --dataset_name SURREAL_64_1-2ev_template_remeshed_augShapes --fmap_direction yx --sample_size 64 --block_out_channels 64,128,128 --down_block_types DownBlock2D,AttnDownBlock2D,AttnDownBlock2D --up_block_types AttnUpBlock2D,AttnUpBlock2D,UpBlock2D

