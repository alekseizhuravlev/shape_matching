#!/bin/bash

#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:8
#SBATCH --partition=mlgpu_medium
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/shape_matching/SLURM_logs/train_diffusion_%j.out
#SBATCH --error=/home/s94zalek_hpc/shape_matching/SLURM_logs/train_diffusion_%j.err

source /home/s94zalek_hpc/.bashrc
conda activate fmnet
cd /home/s94zalek_hpc/shape_matching
module load libGLU Xvfb
export PYTHONPATH=${PYTHONPATH}:/home/s94zalek_hpc/shape_matching


experiment_name='single_64_2-4ev_64-128-128_remeshed_fixed'
dataset_name='SURREAL_64_2-4ev_template_remeshed_augShapes'

fmap_direction='yx'
sample_size=64

block_out_channels='64,128,128'
down_block_types='DownBlock2D,AttnDownBlock2D,AttnDownBlock2D'
up_block_types='AttnUpBlock2D,AttnUpBlock2D,UpBlock2D'


cp -r /lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL/train/${dataset_name} /tmp

srun accelerate launch /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/train_pipeline_distributed.py --experiment_name ${experiment_name} --dataset_name ${dataset_name} --fmap_direction ${fmap_direction} --sample_size ${sample_size} --block_out_channels ${block_out_channels} --down_block_types ${down_block_types} --up_block_types ${up_block_types}

# rewrite the code above to include all arguments in the command line

# cp -r /lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL/train/SURREAL_64_1-2ev_template_remeshed_augShapes /tmp
# accelerate launch /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/train_pipeline_distributed.py --experiment_name single_64_1-2ev_64-128-128_remeshed --dataset_name SURREAL_64_1-2ev_template_remeshed_augShapes --fmap_direction yx --sample_size 64 --block_out_channels 64,128,128 --down_block_types DownBlock2D,AttnDownBlock2D,AttnDownBlock2D --up_block_types AttnUpBlock2D,AttnUpBlock2D,UpBlock2D

