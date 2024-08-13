import networks.diffusion_network as diffusion_network
from tqdm import tqdm
import my_code.sign_canonicalization.training as sign_training
import torch
import my_code.diffusion_training_sign_corr.data_loading as data_loading


def test_on_dataset(net, test_dataset):

    tqdm._instances.clear()

    n_epochs = 5
        
    iterator = tqdm(total=len(test_dataset) * n_epochs)
    incorrect_signs_list = torch.tensor([])
    curr_iter = 0

        
    for _ in range(n_epochs):
        for curr_idx in range(len(test_dataset)):

            ##############################################
            # Select a shape
            ##############################################

            train_shape = test_dataset[curr_idx]['second']

            # train_shape = double_shape['second']
            verts = train_shape['verts'].unsqueeze(0).to(device)
            faces = train_shape['faces'].unsqueeze(0).to(device)    

            evecs_orig = train_shape['evecs'].unsqueeze(0)[:, :, start_dim:start_dim+feature_dim].to(device)
            
            mass_mat = torch.diag_embed(
                torch.ones_like(train_shape['mass'].unsqueeze(0))
                ).to(device)

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
                    net, verts, faces, evecs_flip_0, 
                    mass_mat=mass_mat, input_type=net.input_type,
                    
                    mass=train_shape['mass'].unsqueeze(0), L=train_shape['L'].unsqueeze(0),
                    evals=train_shape['evals'].unsqueeze(0), evecs=train_shape['evecs'].unsqueeze(0),
                    gradX=train_shape['gradX'].unsqueeze(0), gradY=train_shape['gradY'].unsqueeze(0)
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
                    net, verts, faces, evecs_flip_1, 
                    mass_mat=mass_mat, input_type=net.input_type,
                    
                    mass=train_shape['mass'].unsqueeze(0), L=train_shape['L'].unsqueeze(0),
                    evals=train_shape['evals'].unsqueeze(0), evecs=train_shape['evecs'].unsqueeze(0),
                    gradX=train_shape['gradX'].unsqueeze(0), gradY=train_shape['gradY'].unsqueeze(0)
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
        
    return incorrect_signs_list.float().mean(), incorrect_signs_list.max()



if __name__ == '__main__':
        
    condition_dim = 0
    start_dim = 0

    feature_dim = 32
    evecs_per_support = 4


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = diffusion_network.DiffusionNet(
        in_channels=128,
        out_channels=feature_dim // evecs_per_support,
        cache_dir=None,
        input_type='wks',
        k_eig=128,
        n_block=2,
        ).to(device)

    input_type = 'wks'
    
    # exp_name = 'sign_overfit_start_0_inCh_128_feat_32_4block_factor4_dataset_SURREAL_train_rot_180_180_180_normal_True_noise_0.0_-0.05_0.05_lapl_mesh_scale_0.9_1.1_wks'
    exp_name = 'sign_overfit_start_0_inCh_128_iter_20000_feat_32_2block_factor4_dataset_SURREAL_train_rot_180_180_180_normal_True_noise_0.0_-0.05_0.05_lapl_mesh_scale_0.9_1.1_wks'
    
    log_file = f'/home/s94zalek_hpc/shape_matching/my_code/experiments/{exp_name}/log_1.txt'
    

    for dataset_name, split in [
        ('FAUST_a', 'test'),
        ('SHREC19', 'train'), 
        ('FAUST_r', 'test'),
        ('FAUST_orig', 'test'), 
        ('FAUST_r', 'train'), 
        ('FAUST_orig', 'train'), 
        ]:
        
        test_dataset_curr = data_loading.get_val_dataset(
            dataset_name, split, 128, canonicalize_fmap=None, preload=False, return_evecs=True
            )[1]
        
        for n_iter in [2000, 4000, 6000, 8000, 10000, 14000, 20000]:
    
            net.load_state_dict(torch.load(f'/home/s94zalek_hpc/shape_matching/my_code/experiments/{exp_name}/{n_iter}.pth'))

            
            mean_incorrect_signs, max_incorrect_signs = test_on_dataset(net, test_dataset_curr)
            
            print(f'{n_iter}.pth: {dataset_name} {split}: mean {mean_incorrect_signs:.2f} max_incorrect_signs {max_incorrect_signs}')
            
            with open(log_file, 'a') as f:
                f.write(f'{n_iter}.pth: {dataset_name} {split}: mean {mean_incorrect_signs:.2f} max_incorrect_signs {max_incorrect_signs}\n')
            
        with open(log_file, 'a') as f:
                f.write(f'\n')
            

