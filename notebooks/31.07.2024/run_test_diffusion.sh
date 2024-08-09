experiment_name='test_signCorr_withAug_evals_32'

python /home/s94zalek_hpc/shape_matching/notebooks/31.07.2024/test_diffusion_cond.py --experiment_name $experiment_name --dataset_name FAUST_r --split test
python /home/s94zalek_hpc/shape_matching/notebooks/31.07.2024/test_diffusion_cond.py --experiment_name $experiment_name --dataset_name FAUST_orig --split test

python /home/s94zalek_hpc/shape_matching/notebooks/31.07.2024/test_diffusion_cond.py --experiment_name $experiment_name --dataset_name SHREC19 --split train
python /home/s94zalek_hpc/shape_matching/notebooks/31.07.2024/test_diffusion_cond.py --experiment_name $experiment_name --dataset_name SURREAL --split train

python /home/s94zalek_hpc/shape_matching/notebooks/31.07.2024/test_diffusion_cond.py --experiment_name $experiment_name --dataset_name FAUST_a --split test
python /home/s94zalek_hpc/shape_matching/notebooks/31.07.2024/test_diffusion_cond.py --experiment_name $experiment_name --dataset_name FAUST_r --split train
python /home/s94zalek_hpc/shape_matching/notebooks/31.07.2024/test_diffusion_cond.py --experiment_name $experiment_name --dataset_name FAUST_orig --split train
