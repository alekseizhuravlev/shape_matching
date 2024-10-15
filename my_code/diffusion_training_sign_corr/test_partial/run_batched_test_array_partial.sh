experiment_names=(
    'partial_anisRemesh_shot_lambda_0.01_anisRemesh_holes_partial_0.8_xy'
    'partial_anisRemesh_shot_lambda_0.01_anisRemesh_holes_partial_0.8_yx'
    'partial_isoRemesh_shot_lambda_0.01_anisRemesh_holes_partial_0.8_xy'
    'partial_isoRemesh_shot_lambda_0.01_anisRemesh_holes_partial_0.8_yx'
)
checkpoint_names=(
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