import torch
import numpy as np
import matplotlib.pyplot as plt

import trimesh

import my_code.diffusion_training_sign_corr.data_loading as data_loading
import my_code.datasets.shape_dataset as shape_dataset
import my_code.datasets.template_dataset as template_dataset

import networks.diffusion_network as diffusion_network
from tqdm.auto import tqdm
import utils.geometry_util as geometry_util
import robust_laplacian
import scipy.sparse.linalg as sla
import utils.geometry_util as geometry_util
import my_code.sign_canonicalization.training as sign_training


if __name__ == '__main__':

    dataset_single = shape_dataset.SingleFaustDataset(
        phase='train',
        data_root = 'data_with_smpl_corr/FAUST_r',
        centering = 'bbox',
        num_evecs=128,
        # lb_cache_dir=f'/home/s94zalek_hpc/shape_matching/data_with_smpl_corr/FAUST_r/diffusion'
        lb_cache_dir=None
    )

    condition_dim = 0
    start_dim = 0

    feature_dim = 32
    evecs_per_support = 4


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = diffusion_network.DiffusionNet(
        in_channels=feature_dim,
        out_channels=feature_dim // evecs_per_support,
        cache_dir=None,
        input_type='wks',
        k_eig=64,
        n_block=6
        ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    # add scheduler, decay by 0.1 every 30k iterations

    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor=0.1, total_iters=50000)

    input_type = 'wks'
    net.load_state_dict(torch.load('/home/s94zalek_hpc/shape_matching/my_code/experiments/sign_double_start_0_feat_32_6block_factor4_dataset_SURREAL_train_rot_180_180_180_normal_True_noise_0.0_-0.05_0.05_lapl_mesh_scale_0.9_1.1_wks/40000.pth'))

    # tqdm._instances.clear()

    # shapes_to_test = test_shapes
    # net.cache_dir = test_diff_folder

    net.cache_dir = None
            
                
    iterator = tqdm(range(1000))
    incorrect_signs_list = torch.tensor([])
    curr_iter = 0


    import cProfile
    import pandas as pd

    with cProfile.Profile() as pr:
        
        for curr_idx in range(10):     


            ##############################################
            # Select a shape
            ##############################################
            
            test_shape = dataset_single[curr_idx]    
            
            verts = test_shape['verts'].unsqueeze(0).to(device)
            faces = test_shape['faces'].unsqueeze(0).to(device)
            evecs_orig = test_shape['evecs'][:, start_dim:start_dim+feature_dim].unsqueeze(0).to(device)

            ##############################################
            # Set the signs on shape 0
            ##############################################

            # create a random combilation of +1 and -1, length = feature_dim
            sign_gt_0 = torch.randint(0, 2, (feature_dim,)).float().to(device)
            
            sign_gt_0[sign_gt_0 == 0] = -1
            sign_gt_0 = sign_gt_0.float().unsqueeze(0)

            # multiply evecs [6890 x 16] by sign_flip [16]
            evecs_flip_0 = evecs_orig * sign_gt_0
            
            # predict the sign change
            with torch.no_grad():
                sign_pred_0, supp_vec_0, _ = sign_training.predict_sign_change(
                    net, verts, faces, evecs_flip_0, evecs_cond=None, input_type=input_type)
            
            ##############################################
            # Set the signs on shape 1
            ##############################################
            
            # create a random combilation of +1 and -1, length = feature_dim
            sign_gt_1 = torch.randint(0, 2, (feature_dim,)).float().to(device)
            
            sign_gt_1[sign_gt_1 == 0] = -1
            sign_gt_1 = sign_gt_1.float().unsqueeze(0)
            
            # multiply evecs [6890 x 16] by sign_flip [16]
            evecs_flip_1 = evecs_orig * sign_gt_1
            
            # predict the sign change
            with torch.no_grad():
                sign_pred_1, supp_vec_1, _ = sign_training.predict_sign_change(
                    net, verts, faces, evecs_flip_1, evecs_cond=None, input_type=input_type)
            
            ##############################################
            # Calculate the loss
            ##############################################
            
            # calculate the ground truth sign difference
            sign_diff_gt = sign_gt_1 * sign_gt_0
            
            # calculate the sign difference between predicted evecs
            sign_diff_pred = sign_pred_1 * sign_pred_0
            
            sign_correct = sign_diff_pred.sign() * sign_diff_gt.sign() 
            
            
            # count the number of incorrect signs
            count_incorrect_signs = (sign_correct < 0).int().sum()
                
            # incorrect_signs_list.append(count_incorrect_signs)
            incorrect_signs_list = torch.cat([incorrect_signs_list, torch.tensor([count_incorrect_signs])])
            
            
            iterator.set_description(f'Mean incorrect signs {incorrect_signs_list.float().mean():.2f} / {feature_dim}')
            iterator.update(1)
            # if count_incorrect_signs > 7:
            #     raise ValueError('Too many incorrect signs')
        
    df = pd.DataFrame(
        pr.getstats(),
        columns=['func', 'ncalls', 'ccalls', 'tottime', 'cumtime', 'callers']
    )
    # save dataframes to csv
    df.to_csv('/home/s94zalek_hpc/shape_matching/notebooks/31.07.2024/diffusionnet_runtime_profile.csv')