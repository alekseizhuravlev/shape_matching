import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from my_code.datasets.cached.zip_shape_dataset import ZipFileDataset, ZipCollection
from tqdm import tqdm
import networks.diffusion_network as diffusion_network

from tqdm import tqdm
import utils.geometry_util as geometry_util
import robust_laplacian
import scipy.sparse.linalg as sla
import utils.geometry_util as geometry_util
import my_code.sign_canonicalization.training as sign_training





if __name__ == '__main__':
    
    base_dir = '/lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL'
    
    # get all zip files in 
    zip_files_path_list = []
    for file in os.listdir(base_dir)[1:2]:
        if file.endswith('.zip'):
            zip_files_path_list.append(os.path.join(base_dir, file))
            
    zip_files_path_list.sort()

    print(zip_files_path_list)
    
    with ZipCollection(zip_files_path_list) as zip_files:
        dataset = ZipFileDataset(zip_files, 128)        

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
            k_eig=128,
            n_block=6
            ).to(device)
        
        input_type = 'wks'
        # net.load_state_dict(torch.load('/home/s94zalek_hpc/shape_matching/my_code/experiments/sign_double_start_0_feat_32_6block_factor4_dataset_SURREAL_train_rot_180_180_180_normal_True_noise_0.0_-0.05_0.05_lapl_mesh_scale_0.9_1.1_wks/40000.pth'))
        net.load_state_dict(torch.load('/home/s94zalek_hpc/shape_matching/my_code/experiments/sign_estimator_no_aug/40000.pth'))
            
        iterator = tqdm(range(len(dataset)))
        incorrect_signs_list = torch.tensor([])
        curr_iter = 0
            
        for curr_idx in range(len(iterator)):     


            ##############################################
            # Select a shape
            ##############################################
            
            test_shape = dataset[curr_idx]
            
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
                    net, verts, faces, evecs_flip_0, evecs_cond=None, input_type=input_type,
                    mass=test_shape['mass'].unsqueeze(0), L=test_shape['L'].unsqueeze(0),
                    evals=test_shape['evals'], evecs=test_shape['evecs'].unsqueeze(0),
                    gradX=test_shape['gradX'].unsqueeze(0), gradY=test_shape['gradY'].unsqueeze(0)
                    )
            
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
                    net, verts, faces, evecs_flip_1, evecs_cond=None, input_type=input_type,
                    mass=test_shape['mass'].unsqueeze(0), L=test_shape['L'].unsqueeze(0),
                    evals=test_shape['evals'], evecs=test_shape['evecs'].unsqueeze(0),
                    gradX=test_shape['gradX'].unsqueeze(0), gradY=test_shape['gradY'].unsqueeze(0)
                    )
            
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
            
            
            iterator.set_description(f'Mean incorrect signs {incorrect_signs_list.float().mean():.2f} / {feature_dim}, max {incorrect_signs_list.max()}')
            iterator.update(1)
            # if count_incorrect_signs > 7:
            #     raise ValueError('Too many incorrect signs')
        
            
        print(f'Results for {len(incorrect_signs_list)} test shapes')
        print(f'Incorrect signs per shape: {incorrect_signs_list.float().mean():.2f} / {feature_dim}')

        print('Max incorrect signs', incorrect_signs_list.max())
