import torch
import numpy as np
import matplotlib.pyplot as plt
from my_code.datasets.surreal_dataset_3dc import TemplateSurrealDataset3DC
import torch
from tqdm import tqdm
import my_code.sign_canonicalization.training as sign_training    
import networks.diffusion_network as diffusion_network
import time
import os

if __name__ == '__main__':

    if not os.path.exists('/tmp/mmap_datas_surreal_train.pth'):
        
        print('Copying the file to /tmp', flush=True)
        #rsync -aq /home/s94zalek_hpc/3D-CODED/data/mmap_datas_surreal_train.pth /tmp/
        os.system('rsync -aq /lustre/mlnvme/data/s94zalek_hpc-shape_matching/mmap_datas_surreal_train.pth /tmp/')
        
        assert os.path.exists('/tmp/mmap_datas_surreal_train.pth')

    dataset_3dc = TemplateSurrealDataset3DC(
        # shape_path=f'/home/s94zalek_hpc/3D-CODED/data/mmap_datas_surreal_train.pth',
        # shape_path='/lustre/mlnvme/data/s94zalek_hpc-shape_matching/mmap_datas_surreal_train.pth',
        shape_path='/tmp/mmap_datas_surreal_train.pth',
        num_evecs=128,
        use_cuda=False,
        cache_lb_dir=None,
        return_evecs=True,
        mmap=True
    )    

    dataloader_3dc = torch.utils.data.DataLoader(
        dataset_3dc, batch_size=1, shuffle=True,
        num_workers=4,
        # persistent_workers=True,
        pin_memory=False
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
        k_eig=128,
        n_block=6
        ).to(device)
    
    input_type = 'wks'
    chkpt_path = '/home/s94zalek_hpc/shape_matching/my_code/experiments/sign_double_start_0_feat_32_6block_factor4_dataset_SURREAL_train_rot_180_180_180_normal_True_noise_0.0_-0.05_0.05_lapl_mesh_scale_0.9_1.1_wks/40000.pth'
    # chkpt_path = '/home/s94zalek_hpc/shape_matching/my_code/experiments/sign_estimator_no_aug/40000.pth'
    net.load_state_dict(torch.load(chkpt_path))
    # net.load_state_dict(torch.load('/home/s94zalek_hpc/shape_matching/my_code/experiments/sign_estimator_no_aug/40000.pth'))

    # file_name = '/home/s94zalek_hpc/shape_matching/notebooks/31.07.2024/incorrect_signs_noAug.txt'
        
    iterator = tqdm(total=len(dataloader_3dc))
    incorrect_signs_list = torch.tensor([])
 
    curr_iter = 0
    
    for batch in dataloader_3dc:     

        ##############################################
        # Select a shape
        ##############################################
        
        verts = batch['second']['verts'].to(device)
        faces = batch['second']['faces'].to(device)
        evecs_orig = batch['second']['evecs'][:, :, start_dim:start_dim+feature_dim].to(device)

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
                mass=batch['second']['mass'], L=batch['second']['L'],
                evals=batch['second']['evals'], evecs=batch['second']['evecs'],
                gradX=batch['second']['gradX'], gradY=batch['second']['gradY']
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
                mass=batch['second']['mass'], L=batch['second']['L'],
                evals=batch['second']['evals'], evecs=batch['second']['evecs'],
                gradX=batch['second']['gradX'], gradY=batch['second']['gradY']
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
        
        # if curr_iter == 0:
        #     with open(file_name, 'w') as f:
        #         f.write('Incorrect signs per shape\n')
        #         f.write(f'{chkpt_path}\n')
        # elif curr_iter % 25 == 0:
        #     with open(file_name, 'a') as f:
        #         f.write(f'{curr_iter:06d} Mean {incorrect_signs_list.float().mean():.2f} / {feature_dim}, max {incorrect_signs_list.max()}\n')
        
        if curr_iter % 100 == 0:
            os.system('ps -v')
            os.system('free --giga')
        
        curr_iter += 1
        
        
        
        # if count_incorrect_signs > 7:
        #     raise ValueError('Too many incorrect signs')
        
        # time.sleep(0.25)
        
        
        
    print(f'Results for {len(incorrect_signs_list)} test shapes')
    print(f'Incorrect signs per shape: {incorrect_signs_list.float().mean():.2f} / {feature_dim}')

    print('Max incorrect signs', incorrect_signs_list.max())