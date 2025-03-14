#!/bin/bash

#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --partition=mlgpu_medium
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/shape_matching/SLURM_logs/test_sign_corr_%A_%a.out
#SBATCH --error=/home/s94zalek_hpc/shape_matching/SLURM_logs/test_sign_corr_%A_%a.err

source /home/s94zalek_hpc/.bashrc
conda activate fmnet
cd /home/s94zalek_hpc/shape_matching
module load libGLU Xvfb
export PYTHONPATH=${PYTHONPATH}:/home/s94zalek_hpc/shape_matching

# put all dataset names and splits in a list
job_list=(
    # 'signNet_128_remeshed_mass_6b_1-1-2-2ev_10_0.2_0.8'
    # 'signNet_128_remeshed_mass_6b_1-2-2-2ev_10_0.2_0.8'
    # 'signNet_128_remeshed_mass_6b_1-2-2-4ev_10_0.2_0.8'

    # 'signNet_remeshed_mass_6b_1ev_10_0.2_0.8'
    # 'signNet_24_remeshed_mass_6b_1ev_10_0.2_0.8'
    # 'signNet_48_remeshed_mass_6b_1ev_10_0.2_0.8'
    # 'signNet_64_remeshed_mass_6b_1ev_10_0.2_0.8'

    'signNet_remeshed_mass_6b_1ev_10_0.2_0.8'
    'signNet_64_remeshed_mass_6b_1-2ev_10_0.2_0.8'
    'signNet_96_remeshed_mass_6b_1-2-4ev_10_0.2_0.8'


    # 'signNet_96_remeshed_mass_6b_1-2-2ev_10_0.2_0.8'
    # 'signNet_128_remeshed_mass_6b_1-2-2-4ev_10_0.2_0.8'
    # 'signNet_128_remeshed_mass_6b_1-2-4-4ev_10_0.2_0.8'
    # 'signNet_128_remeshed_mass_6b_1-2-4-8ev_10_0.2_0.8'
)

partial=-1

# worker id = id of the current job in the job list
worker_id=$SLURM_ARRAY_TASK_ID

# get the current job from the job list
exp_name=${job_list[$worker_id]}
echo "Running job $worker_id: exp_name=$exp_name"

# run the job
srun python /home/s94zalek_hpc/shape_matching/my_code/sign_canonicalization/test_sign_correction.py --exp_name $exp_name --remesh_targetlen 0 --smoothing_iter 0 --partial $partial
# srun python /home/s94zalek_hpc/shape_matching/my_code/sign_canonicalization/test_sign_correction.py --exp_name $exp_name --remesh_targetlen 1 --smoothing_type taubin --smoothing_iter 5 --partial $partial




