experiment_names=(
    'partial_0.8_5k_32_1_lambda_0.01_anisRemesh_cuts_xy'
    'partial_0.8_5k_32_1_lambda_0.01_anisRemesh_cuts_yx'
    'partial_0.8_5k_32_1_lambda_0.001_anisRemesh_cuts_xy'
    'partial_0.8_5k_32_1_lambda_0.01_xy'
    'partial_0.8_5k_32_2_lambda_0.01_xy'
    'partial_0.8_5k_xyz_32_1_lambda_0.01_anisRemesh_holes_bbox_partial_0.8_xy'
    'partial_0.8_5k_xyz_32_1_lambda_0.01_anisRemesh_cuts_bbox_partial_0.8_xy'
    'partial_0.8_5k_xyz_32_2_-1_anisRemesh_cuts_bbox_partial_0.8_xy'
)
checkpoint_names=(
    'epoch_95'
    'epoch_95'
    'epoch_95'
    'epoch_95'
    'epoch_95'
    'epoch_95'
    'epoch_95'
    'epoch_95'
)

# run sbatch /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/run_test_diffusion_array.sh with each experiment name
# for experiment_name in "${experiment_names[@]}"
# do
#     sbatch /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/run_test_diffusion_array.sh $experiment_name
# done

# run sbatch /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/run_test_diffusion_array.sh with each experiment name and checkpoint name
for i in "${!experiment_names[@]}"
do
    experiment_name=${experiment_names[$i]}
    checkpoint_name=${checkpoint_names[$i]}
    sbatch /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test_partial/run_test_diffusion_array_partial.sh $experiment_name $checkpoint_name
done