import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import trimesh
import my_code.diffusion_training.data_loading as data_loading
import my_code.diffusion_training.evaluate_samples as evaluate_samples

from tqdm import tqdm
import utils.geometry_util as geometry_util

from pyFM_fork.pyFM.refine.zoomout import zoomout_refine



def flip_random_signs(
    C_xy, idx_start, idx_end, n_signs, row_wise
):
    C_xy = C_xy.clone()
    
    # randomly sample n non-repeating indices between idx_start and idx_end
    idx = np.random.choice(
        np.arange(idx_start, idx_end), n_signs, replace=False
    )
    idx = torch.tensor(idx)
    
    # print(idx)
    
    if row_wise:
        C_xy[idx] = -C_xy[idx]
    else:
        C_xy[:, idx] = -C_xy[:, idx]

    return C_xy



if __name__ == '__main__':

    split_name = 'train'
    save_folder = f'/home/s94zalek_hpc/shape_matching/my_code/experiments/zoomout_sign_error_to_p2p_{split_name}'
    os.makedirs(save_folder, exist_ok=True)
    
    test_dataset_128 = data_loading.get_val_dataset(
        'FAUST_orig', split_name, 128, canonicalize_fmap=None
        )[1]
    
    factor_increase_per_sign = []
    geo_err_gt_per_sign = []
    geo_err_est_per_sign = []

    n_sign_flips_list = [1, 2, 4]
    for n_signs in n_sign_flips_list:
        
        factor_increase_list = torch.tensor([])
        geo_err_gt_list = torch.tensor([])
        geo_err_est_list = torch.tensor([])  
        
        iterator = tqdm(range(500))
        for _ in iterator:
            
            # select a random shape
            rand_idx = np.random.randint(1, len(test_dataset_128))
            data_10 = test_dataset_128[rand_idx]
            
            # get the ground truth C_xy
            C_gt_xy = data_10['second']['C_gt_xy'][0, :64, :64]
            
            # randomly flip the signs
            C_xy_err_unref = flip_random_signs(
                C_gt_xy,
                idx_start=0,
                idx_end=64,
                n_signs=n_signs,
                row_wise=True
            )
            
            C_xy_err = zoomout_refine(
                FM_12=C_xy_err_unref.numpy(), 
                evects1=data_10['first']['evecs'].numpy(), 
                evects2=data_10['second']['evecs'].numpy(),
                nit=8, step=8,
                verbose=False
            )
            C_xy_err = torch.tensor(C_xy_err)
            
            
            # calculate the geodesic error
            geo_err_gt, geo_err_est, p2p_gt, p2p_est = evaluate_samples.compare_fmap_with_gt(
                Cxy_est=C_xy_err.unsqueeze(0),
                data_x=data_10['first'],
                data_y=data_10['second']
            )
            factor_increase = geo_err_est.sum() / geo_err_gt.sum()
            factor_increase_list = torch.cat([factor_increase_list, factor_increase.unsqueeze(0)])
            
            mean_gt = geo_err_gt.mean().unsqueeze(0) * 100
            mean_est = geo_err_est.mean().unsqueeze(0) * 100
            
            geo_err_gt_list = torch.cat([geo_err_gt_list, mean_gt])
            geo_err_est_list = torch.cat([geo_err_est_list, mean_est])
            
            # if factor_increase > 3:
            #     iterator.set_description(f'factor_increase {factor_increase:.2f},shape {rand_idx}')
            
            #'factor_increase mean {factor_increase_list.mean():.3f}'
            iterator.set_description(f'geo_err_est mean {geo_err_est_list.mean():.3f}')

        # mean_factor_increase_list.append(factor_increase_list.mean())
        # median_factor_increase_list.append(factor_increase_list.median())
        # max_factor_increase_list.append(factor_increase_list.max())
        # min_factor_increase_list.append(factor_increase_list.min())
        
        factor_increase_per_sign.append(factor_increase_list)
        geo_err_gt_per_sign.append(geo_err_gt_list)
        geo_err_est_per_sign.append(geo_err_est_list)
        
        print(f'{n_signs} signs:')    
        
        print(f'factor_increase mean {factor_increase_list.mean():.3f}')
        print(f'median {factor_increase_list.median():.3f}')
        print(f'max {factor_increase_list.max():.3f}')
        print(f'min {factor_increase_list.min():.3f}')
        print()
        
        print(f'geo_err_est mean {geo_err_est_list.mean():.2f}')
        print(f'median {geo_err_est_list.median():.2f}')
        print(f'max {geo_err_est_list.max():.2f}')
        print(f'min {geo_err_est_list.min():.2f}')
        print()
    

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        axs[0].boxplot(geo_err_est_per_sign, showfliers=False)
        axs[0].set_title('Geo error est, %')
        axs[0].set_xlabel('Number of signs flipped')
        axs[0].set_ylabel('Geo error est, %')
        axs[0].set_xticklabels(n_sign_flips_list[:len(geo_err_est_per_sign)])

        # draw a horizontal line at 1.6
        axs[0].axhline(y=1.6, color='r', linestyle='--')

        # do not show outliers
        axs[1].boxplot(factor_increase_per_sign, showfliers=False)
        axs[1].set_title('Factor increase vs GT')
        axs[1].set_xlabel('Number of signs flipped')
        axs[1].set_ylabel('Factor increase')
        axs[1].set_xticklabels(n_sign_flips_list[:len(geo_err_est_per_sign)])

        # save the figure to /home/s94zalek_hpc/shape_matching/my_code/experiments/zoomout_sign_error_to_p2p
        fig.savefig(f'{save_folder}/boxplot_{n_signs}_signs.png')
        plt.close(fig)