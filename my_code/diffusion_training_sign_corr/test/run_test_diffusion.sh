experiment_name='pair_augShapes_signNet_remeshed_4b_mass_10_0.2_0.8'
checkpoint_name='checkpoint_99.pt'

# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name FAUST_orig --split test
# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name FAUST_orig_pair --split test

# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name FAUST_r --split test
# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name FAUST_r_pair --split test

# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name FAUST_a --split test
# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name FAUST_a_pair --split test

python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name SHREC19_r_pair --split test

# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name SCAPE_r_pair --split test
# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name SCAPE_a_pair --split test


# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name FAUST_r --split train
# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name FAUST_orig --split train

# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond.py --experiment_name $experiment_name --checkpoint_name $checkpoint_name --dataset_name SURREAL --split train