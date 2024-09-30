experiment_names=(
    # 'single_24_remeshed_noAcc_yx'
    'single_template_remeshed'
    'single_48_remeshed_noAcc_yx_64_128_128'
    'single_64_1-2ev_64-128-128_remeshed_fixed'
    'single_64_1-4ev_64-128-128_remeshed_fixed'
    'single_64_2-2ev_64-128-128_remeshed_fixed'
    'single_64_2-4ev_64-128-128_remeshed_fixed'
)
checkpoint_names=(
    # 'checkpoint_99.pt'
    'checkpoint_99.pt'
    'checkpoint_95.pt'
    'epoch_99'
    'epoch_99'
    'epoch_99'
    'epoch_99'
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
    sbatch /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/run_test_diffusion_array.sh $experiment_name $checkpoint_name
done