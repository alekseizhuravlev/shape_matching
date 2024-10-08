import os
import networks.diffusion_network as diffusion_network
from tqdm import tqdm
import my_code.sign_canonicalization.training as sign_training
import my_code.sign_canonicalization.remesh as remesh
import torch
import my_code.diffusion_training_sign_corr.data_loading as data_loading
import yaml
import my_code.datasets.preprocessing as preprocessing
import trimesh
import argparse
import utils.fmap_util as fmap_util
    


def remesh_dataset(dataset, name, remesh_targetlen, smoothing_iter, smoothing_type, num_evecs):

    new_dataset = []

    for i in tqdm(range(len(dataset)), desc=f'Remeshing the dataset {name}'):
        
        train_shape_orig = dataset[i]

        verts_orig = train_shape_orig['verts']
        faces_orig = train_shape_orig['faces']
        
        verts, faces = remesh.remesh_simplify_iso(
            verts_orig,
            faces_orig,
            n_remesh_iters=10,
            remesh_targetlen=remesh_targetlen,
            simplify_strength=1,
        )
        
        mesh_anis_remeshed = trimesh.Trimesh(verts, faces, process=False)
        # apply laplacian smoothing
        
        if smoothing_type == 'laplacian':
            trimesh.smoothing.filter_laplacian(mesh_anis_remeshed, lamb=0.5, iterations=smoothing_iter)
        elif smoothing_type == 'taubin':
            trimesh.smoothing.filter_taubin(mesh_anis_remeshed, lamb=0.5, iterations=smoothing_iter)
        else:
            raise ValueError(f'Unknown smoothing type {smoothing_type}')
            
        # trimesh.smoothing.filter_laplacian(mesh_anis_remeshed, lamb=0.5, iterations=smoothing_iter)
        # trimesh.smoothing.filter_taubin(mesh_anis_remeshed, lamb=0.5, iterations=smoothing_iter)
        
        corr_orig_to_remeshed = fmap_util.nn_query(
            verts,
            verts_orig, 
            )
        
        train_shape = {
            'verts_orig': verts_orig,
            'faces_orig': faces_orig,
            'corr_orig_to_remeshed': corr_orig_to_remeshed,
            'verts': torch.tensor(mesh_anis_remeshed.vertices).float(),
            'faces': torch.tensor(mesh_anis_remeshed.faces).int(),
        }
        train_shape = preprocessing.get_spectral_ops(train_shape, num_evecs=num_evecs,
                                    cache_dir=None)
        
        new_dataset.append(train_shape)
    
    return new_dataset


def test_on_dataset(net, test_dataset, with_mass, n_epochs):

    tqdm._instances.clear()
        
    iterator = tqdm(total=len(test_dataset) * n_epochs)
    incorrect_signs_list = torch.tensor([])
    curr_iter = 0

        
    for _ in range(n_epochs):
        for curr_idx in range(len(test_dataset)):

            ##############################################
            # Select a shape
            ##############################################

            train_shape = test_dataset[curr_idx]           
            
            ##############################################
            # Set the variables
            ##############################################

            # train_shape = double_shape['second']
            verts = train_shape['verts'].unsqueeze(0).to(device)
            faces = train_shape['faces'].unsqueeze(0).to(device)    

            evecs_orig = train_shape['evecs'].unsqueeze(0)[:, :, start_dim:start_dim+feature_dim].to(device)
            
            if with_mass:
                mass_mat = torch.diag_embed(
                    train_shape['mass'].unsqueeze(0)
                    ).to(device)
            else:
                mass_mat = None

            ##############################################
            # Set the signs on shape 0
            ##############################################

            # create a random combilation of +1 and -1, length = feature_dim
            sign_gt_0 = torch.randint(0, 2, (feature_dim,)).float().to(device)
            
            sign_gt_0[sign_gt_0 == 0] = -1
            sign_gt_0 = sign_gt_0.float().unsqueeze(0)

            # print('evecs_orig', evecs_orig.shape, 'sign_gt_0', sign_gt_0.shape)

            # multiply evecs [6890 x 16] by sign_flip [16]
            evecs_flip_0 = evecs_orig * sign_gt_0
            
            
            
            # predict the sign change
            with torch.no_grad():
                sign_pred_0, supp_vec_0, _ = sign_training.predict_sign_change(
                    net, verts, faces, evecs_flip_0, 
                    mass_mat=mass_mat, input_type=net.input_type,
                    evecs_per_support=config['evecs_per_support'],
                    
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
                    evecs_per_support=config['evecs_per_support'],
                    
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
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--remesh_targetlen', type=float)
    parser.add_argument('--smoothing_type', 
                        choices=['laplacian', 'taubin'])
                        
    parser.add_argument('--smoothing_iter', type=int)
    
    parser.add_argument('--partial', type=bool, required=True)
    
    args = parser.parse_args()
    
    exp_name = args.exp_name
    remesh_targetlen = args.remesh_targetlen  
    smoothing_type = args.smoothing_type  
    smoothing_iter = args.smoothing_iter
        
    # exp_name = 'signNet_remeshed_4b_mass_10_0.2_0.8' 
    # exp_name = 'signNet_orig' 
    # exp_name = 'signNet_remeshed_10_0.5_1' 
    
    
    exp_dir = f'/home/s94zalek_hpc/shape_matching/my_code/experiments/sign_net/{exp_name}'
    
    with open(f'{exp_dir}/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # remesh_targetlen = None
    # smoothing_iter = 0
    # remesh_targetlen = config['dataset']['remesh']['isotropic']['remesh_targetlen']
        
    start_dim = config['start_dim']

    feature_dim = config['feature_dim']
    evecs_per_support = config['evecs_per_support']


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = diffusion_network.DiffusionNet(
        **config['net_params']
        ).to(device)

    input_type = config['net_params']['input_type']
    
    
    log_file = f'{exp_dir}/log_10ep_remesh_{remesh_targetlen}_smooth_{smoothing_type}_{smoothing_iter}.txt'
    # log_file = f'{exp_dir}/log_10ep_remesh_{remesh_targetlen}_laplacianSmooth_{smoothing_iter}.txt'
    # log_file = f'{exp_dir}/log_10ep_noRemesh_Shrec19R.txt'
    
    # log_file = f'{exp_dir}/logs_shrec/log_10ep_laplacianSmooth_{smoothing_iter}.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    
    if args.partial:
        dataset_list = [
            ('SHREC16_cuts_pair', 'test'),
            ('SHREC16_holes_pair', 'test'),
            ('FAUST_r', 'test'),
            ('FAUST_orig', 'test'), 
            ('FAUST_a', 'test'),
            ('SCAPE_r_pair', 'test'),
            ('SCAPE_a_pair', 'test'),
            ('SHREC19_r', 'train'), 
            
        ]
    else:    
        dataset_list = [
            # (config["train_folder"], 'train'),
            
            ('FAUST_a', 'test'),
            ('SHREC19_r', 'train'), 
            ('FAUST_r', 'test'),
            ('FAUST_orig', 'test'), 
            ('FAUST_r', 'train'), 
            ('FAUST_orig', 'train'), 
            ('SCAPE_r_pair', 'test'),
            ('SCAPE_a_pair', 'test'),
            ('SCAPE_r_pair', 'train'),
            
            # ('DT4D_intra_pair', 'test'),
            # ('DT4D_intra_pair', 'train'),
            # ('DT4D_inter_pair', 'test'),
            # ('DT4D_inter_pair', 'train'),
        ]
    

    for n_iter in [50000]:
    # for n_iter in [200, 600, 1000, 1400, 2000]:

        net.load_state_dict(torch.load(f'{exp_dir}/{n_iter}.pth'))


        for dataset_name, split in dataset_list:
            
            if dataset_name == config["train_folder"]:
                test_dataset_curr, _ = sign_training.load_cached_shapes(
                    f'/home/s94zalek_hpc/shape_matching/data_sign_training/train/{config["train_folder"]}',
                    unsqueeze=False
                )  

                    
                mean_incorrect_signs, max_incorrect_signs = test_on_dataset(net, test_dataset_curr, with_mass=config['with_mass'], n_epochs=1)
                
            else:
                test_dataset_curr = data_loading.get_val_dataset(
                    dataset_name, split, 128, canonicalize_fmap=None, preload=False, return_evecs=True
                    )[0]
                
                if remesh_targetlen is not None and remesh_targetlen > 0:
                    test_dataset_curr = remesh_dataset(
                        test_dataset_curr, dataset_name,
                        remesh_targetlen, num_evecs=net.k_eig,
                        smoothing_iter=smoothing_iter,
                        smoothing_type=smoothing_type
                        )
                
                mean_incorrect_signs, max_incorrect_signs = test_on_dataset(
                    net, test_dataset_curr,
                    with_mass=config['with_mass'], n_epochs=25)
    
            
            print(f'{n_iter}.pth: {dataset_name} {split}: mean {mean_incorrect_signs * 100 / feature_dim:.2f}% max_incorrect_signs {max_incorrect_signs * 100 / feature_dim:.2f}% (Mean {mean_incorrect_signs:.2f} / {feature_dim} Max {max_incorrect_signs})')
            
            with open(log_file, 'a') as f:
                f.write(f'{n_iter}.pth: {dataset_name} {split}: mean {mean_incorrect_signs * 100 / feature_dim:.2f}% max_incorrect_signs {max_incorrect_signs * 100 / feature_dim:.2f}% (Mean {mean_incorrect_signs:.2f} Max {max_incorrect_signs})\n')
            
        with open(log_file, 'a') as f:
                f.write(f'\n')
                
    with open(log_file, 'a') as f:
        f.write(f'\n')
            

