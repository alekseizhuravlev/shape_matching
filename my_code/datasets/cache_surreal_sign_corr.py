import os
import numpy as np
import shutil

import torch
from tqdm import tqdm
import argparse
import time

import sys
import os

import yaml
curr_dir = os.getcwd()
if 's94zalek_hpc' in curr_dir:
    user_name = 's94zalek_hpc'
else:
    user_name = 's94zalek'
sys.path.append(f'/home/{user_name}/shape_matching/')

from my_code.sign_canonicalization.training import predict_sign_change
import networks.diffusion_network as diffusion_network
import my_code.utils.plotting_utils as plotting_utils
import matplotlib.pyplot as plt

import my_code.datasets.shape_dataset as shape_dataset
import my_code.datasets.template_dataset as template_dataset
from my_code.datasets.surreal_dataset_3dc import TemplateSurrealDataset3DC
from my_code.diffusion_training_sign_corr.test.test_diffusion_pair_template_unified import RegularizedFMNet    
    
def visualize_before_after(data, C_xy_corr, C_yx_corr, evecs_cond_first, evecs_cond_second, figures_folder, idx):
        l = 0
        h = num_evecs

        fig, axs = plt.subplots(1, 6, figsize=(18, 5))

        plotting_utils.plot_Cxy(fig, axs[0], data['second']['C_gt_xy'],
                                'before', l, h, show_grid=False, show_colorbar=False)
        plotting_utils.plot_Cxy(fig, axs[1], C_xy_corr,
                                'after', l, h, show_grid=False, show_colorbar=False)
        plotting_utils.plot_Cxy(fig, axs[2], C_yx_corr,
                                'C_yx', l, h, show_grid=False, show_colorbar=False)
        # plotting_utils.plot_Cxy(fig, axs[3], data['second']['C_gt_xy'][l:h, l:h] - C_xy_corr,
        #                 'diff', l, h, show_grid=False, show_colorbar=False)
        plotting_utils.plot_Cxy(fig, axs[3], data['second']['C_gt_xy'][l:h, l:h].abs() - C_xy_corr.abs(),
                        'abs diff', l, h, show_grid=False, show_colorbar=False)
        plotting_utils.plot_Cxy(fig, axs[4], evecs_cond_first,
                        'evecs_cond_first', l, h, show_grid=False, show_colorbar=False)
        plotting_utils.plot_Cxy(fig, axs[5], evecs_cond_second,
                        'evecs_cond_second', l, h, show_grid=False, show_colorbar=False)

        # save the figure
        fig.savefig(f'{figures_folder}/{idx}.png')
        plt.close(fig)
        
    
def get_corrected_data(data, num_evecs, net, net_input_type, with_mass, evecs_per_support, fmnet):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    
    verts_first = data['first']['verts'].unsqueeze(0).to(device)
    verts_second = data['second']['verts'].unsqueeze(0).to(device)
    
    faces_first = data['first']['faces'].unsqueeze(0).to(device)
    faces_second = data['second']['faces'].unsqueeze(0).to(device)

    evecs_first = data['first']['evecs'][:, :num_evecs].unsqueeze(0).to(device)
    evecs_second = data['second']['evecs'][:, :num_evecs].unsqueeze(0).to(device)

    corr_first = data['first']['corr']
    corr_second = data['second']['corr']
    
    if net_input_type == 'shot':
        input_feats_first = data['first']['shot'].unsqueeze(0).to(device)    
        input_feats_second = data['second']['shot'].unsqueeze(0).to(device)
    else:
        input_feats_first = None
        input_feats_second = None
        
    
    
    
    if with_mass:
        mass_mat_first = torch.diag_embed(
            data['first']['mass'].unsqueeze(0)
            ).to(device)
        mass_mat_second = torch.diag_embed(
            data['second']['mass'].unsqueeze(0)
            ).to(device)
    else:
        mass_mat_first = None
        mass_mat_second = None


    # predict the sign change
    with torch.no_grad():
        sign_pred_first, support_vector_norm_first, _ = predict_sign_change(
            net, verts_first, faces_first, evecs_first, 
            mass_mat=mass_mat_first, input_type=net_input_type,
            evecs_per_support=evecs_per_support,
            
            input_feats=input_feats_first,
            
            # mass=None, L=None, evals=None, evecs=None, gradX=None, gradY=None
            mass=data['first']['mass'].unsqueeze(0), L=data['first']['L'].unsqueeze(0),
            evals=data['first']['evals'].unsqueeze(0), evecs=data['first']['evecs'].unsqueeze(0),
            gradX=data['first']['gradX'].unsqueeze(0), gradY=data['first']['gradY'].unsqueeze(0)
            )
        sign_pred_second, support_vector_norm_second, _ = predict_sign_change(
            net, verts_second, faces_second, evecs_second, 
            mass_mat=mass_mat_second, input_type=net_input_type,
            evecs_per_support=evecs_per_support,
            
            input_feats=input_feats_second,
            
            # mass=None, L=None, evals=None, evecs=None, gradX=None, gradY=None
            mass=data['second']['mass'].unsqueeze(0), L=data['second']['L'].unsqueeze(0),
            evals=data['second']['evals'].unsqueeze(0), evecs=data['second']['evecs'].unsqueeze(0),
            gradX=data['second']['gradX'].unsqueeze(0), gradY=data['second']['gradY'].unsqueeze(0)
            )

    # correct the evecs
    evecs_first_corrected = evecs_first[0] * torch.sign(sign_pred_first)
    # evecs_first_corrected_norm = evecs_first_corrected / torch.norm(evecs_first_corrected, dim=0, keepdim=True)
    evecs_first_corrected_norm = torch.nn.functional.normalize(evecs_first_corrected, p=2, dim=0)
    
    evecs_second_corrected = evecs_second[0] * torch.sign(sign_pred_second)
    # evecs_second_corrected_norm = evecs_second_corrected / torch.norm(evecs_second_corrected, dim=0, keepdim=True)
    evecs_second_corrected_norm = torch.nn.functional.normalize(evecs_second_corrected, p=2, dim=0)
    
    
    # product with support
    if with_mass:
        evecs_cond_first = torch.nn.functional.normalize(
            support_vector_norm_first[0].transpose(0, 1) \
                @ mass_mat_first[0],
            p=2, dim=1) \
                @ evecs_first_corrected_norm
        
        evecs_cond_second = torch.nn.functional.normalize(
            support_vector_norm_second[0].transpose(0, 1) \
                @ mass_mat_second[0],
            p=2, dim=1) \
                @ evecs_second_corrected_norm 
        
    else:
        evecs_cond_first = support_vector_norm_first[0].transpose(0, 1) @ evecs_first_corrected_norm
        evecs_cond_second = support_vector_norm_second[0].transpose(0, 1) @ evecs_second_corrected_norm
    
    # wrong order?
    # evecs_cond_first = evecs_first_corrected.transpose(0, 1) @ support_vector_norm_first[0].cpu()
    # evecs_cond_second = evecs_second_corrected.transpose(0, 1) @ support_vector_norm_second[0].cpu()


    if fmnet is None:
        # correct the functional map
        C_xy_pred = torch.linalg.lstsq(
            evecs_second_corrected[corr_second],
            evecs_first_corrected[corr_first],
            ).solution
        
        C_yx_pred = torch.linalg.lstsq(
            evecs_first_corrected[corr_first],
            evecs_second_corrected[corr_second],
            ).solution
        
    else:
        # regularized version
        evecs_trans_first = evecs_first_corrected.T * data['first']['mass'][None].to(device)
        evecs_trans_second = evecs_second_corrected.T * data['second']['mass'][None].to(device)
        
        C_xy_pred = fmnet.compute_functional_map(
            evecs_trans_second[:, corr_second].unsqueeze(0).to(device),
            evecs_trans_first[:, corr_first].unsqueeze(0).to(device),
            data['second']['evals'][:num_evecs].unsqueeze(0).to(device),
            data['first']['evals'][:num_evecs].unsqueeze(0).to(device),
        )[0].T

        C_yx_pred = fmnet.compute_functional_map(
            evecs_trans_first[:, corr_first].unsqueeze(0).to(device),
            evecs_trans_second[:, corr_second].unsqueeze(0).to(device),
            data['first']['evals'][:num_evecs].unsqueeze(0).to(device),
            data['second']['evals'][:num_evecs].unsqueeze(0).to(device),
        )[0].T
        

    return C_xy_pred.cpu(), C_yx_pred.cpu(), evecs_cond_first.cpu(), evecs_cond_second.cpu(), \
        evecs_first_corrected.cpu(), evecs_second_corrected.cpu(), \
            corr_first.cpu(), corr_second.cpu()
    
    
    
def save_train_dataset(
        dataset,
        train_indices,
        dataset_folder,
        start_idx,
        end_idx,
        num_evecs,
        pair_type,
        n_pairs,
        fmnet,
        **net_params
    ):
    
    curr_time = time.time()
    
    train_folder = f'{dataset_folder}/train'
    os.makedirs(train_folder, exist_ok=True)
    
    figures_folder = f'{train_folder}/figures'
    os.makedirs(figures_folder, exist_ok=True)

    # evals_first_file = os.path.join(train_folder, f'evals_first_{start_idx}_{end_idx}.pt')
    # evals_second_file = os.path.join(train_folder, f'evals_second_{start_idx}_{end_idx}.pt')
    # fmaps_xy_file = os.path.join(train_folder, f'C_gt_xy_{start_idx}_{end_idx}.pt')
    # fmaps_yx_file = os.path.join(train_folder, f'C_gt_yx_{start_idx}_{end_idx}.pt')
    # evecs_cond_first_file = os.path.join(train_folder, f'evecs_cond_first_{start_idx}_{end_idx}.pt')
    # evecs_cond_second_file = os.path.join(train_folder, f'evecs_cond_second_{start_idx}_{end_idx}.pt')
    
    # print(f'Saving evals to {evals_second_file}', f'fmaps to {fmaps_xy_file}', f'evecs_cond to {evecs_cond_first_file}')
    
    evals_first_tensor = torch.tensor([])
    evals_second_tensor = torch.tensor([])
    fmaps_xy_tensor = torch.tensor([])
    fmaps_yx_tensor = torch.tensor([])
    evecs_cond_first_tensor = torch.tensor([])
    evecs_cond_second_tensor = torch.tensor([])
    
    # evecs_first_corrected_tensor = torch.tensor([])
    # evecs_second_corrected_tensor = torch.tensor([])
    # corr_first_tensor = torch.tensor([])
    # corr_second_tensor = torch.tensor([])
    
    evecs_first_with_corr_tensor = torch.tensor([])
    evecs_second_with_corr_tensor = torch.tensor([])


    for i, idx in enumerate(train_indices):
           
        data_first_idx = dataset[idx]
        second_indices = set()
        
        for curr_pair_j in range(n_pairs):
                    
            if pair_type == 'template':
                # first shape is the template
                data = data_first_idx
                
            elif pair_type == 'pair':

                # select the second shape
                second_idx = np.random.randint(len(dataset))
                
                # second shape != first and wasn't selected before
                while second_idx == idx or second_idx in second_indices:
                    second_idx = np.random.randint(len(dataset))
                second_indices.add(second_idx)
                    
                # get the second shape and fill the data
                data_second_idx = dataset[second_idx]
                data = {
                    'first': data_first_idx['second'],
                    'second': data_second_idx['second']
                }
                
            # get the evals
            evals_first = data['first']['evals'][:num_evecs]
            evals_second = data['second']['evals'][:num_evecs]
            
            # correct the evecs and fmaps
            C_xy_corr, C_yx_corr, evecs_cond_first, evecs_cond_second, evecs_first_corrected, evecs_second_corrected, corr_first, corr_second = get_corrected_data(
                data=data,
                num_evecs=num_evecs,
                fmnet=fmnet,
                **net_params
            )
            # assert the tensors have the correct shapes
            assert C_xy_corr.shape == (num_evecs, num_evecs) and C_yx_corr.shape == (num_evecs, num_evecs), f'{C_xy_corr.shape}, {C_yx_corr.shape}'
            assert evecs_cond_first.shape == (num_evecs, num_evecs) and evecs_cond_second.shape == (num_evecs, num_evecs), f'{evecs_cond_first.shape}, {evecs_cond_second.shape}'
            assert evals_first.shape == (num_evecs,) and evals_second.shape == (num_evecs,), f'{evals_first.shape}, {evals_second.shape}'

            # append to the tensors
            evals_first_tensor = torch.cat((evals_first_tensor, evals_first.unsqueeze(0)), dim=0)
            evals_second_tensor = torch.cat((evals_second_tensor, evals_second.unsqueeze(0)), dim=0)
            fmaps_xy_tensor = torch.cat((fmaps_xy_tensor, C_xy_corr.unsqueeze(0)), dim=0)
            fmaps_yx_tensor = torch.cat((fmaps_yx_tensor, C_yx_corr.unsqueeze(0)), dim=0)
            evecs_cond_first_tensor = torch.cat((evecs_cond_first_tensor, evecs_cond_first.unsqueeze(0)), dim=0)
            evecs_cond_second_tensor = torch.cat((evecs_cond_second_tensor, evecs_cond_second.unsqueeze(0)), dim=0)
        
        
            if pair_type == 'pair':
                evecs_first_with_corr = evecs_first_corrected[corr_first]
                evecs_second_with_corr = evecs_second_corrected[corr_second]
                
                evecs_first_with_corr_tensor = torch.cat((evecs_first_with_corr_tensor, evecs_first_with_corr.unsqueeze(0)), dim=0)
                evecs_second_with_corr_tensor = torch.cat((evecs_second_with_corr_tensor, evecs_second_with_corr.unsqueeze(0)), dim=0)
        
        
        
            
        if pair_type == 'pair' or i % 100 == 0 or i == 15:
            time_elapsed = time.time() - curr_time
            print(f'{i}/{len(train_indices)}, time: {time_elapsed:.2f}, avg: {time_elapsed / (i + 1):.2f}, second_indices: {second_indices}',
                flush=True)
            
        if i < 5 or i % 1000 == 0:
        # if i % 1000 == 0:
            data['second']['C_gt_xy'], data['second']['C_gt_yx'] =\
                dataset.get_functional_map(data['first'], data['second'])
            
            visualize_before_after(
                data, C_xy_corr, C_yx_corr, 
                evecs_cond_first, evecs_cond_second,
                figures_folder, idx)
            
    torch.save(evals_first_tensor, f'{train_folder}/evals_first_{start_idx}_{end_idx}.pt')
    torch.save(evals_second_tensor, f'{train_folder}/evals_second_{start_idx}_{end_idx}.pt')
    torch.save(fmaps_xy_tensor, f'{train_folder}/C_gt_xy_{start_idx}_{end_idx}.pt')
    torch.save(fmaps_yx_tensor, f'{train_folder}/C_gt_yx_{start_idx}_{end_idx}.pt')
    torch.save(evecs_cond_first_tensor, f'{train_folder}/evecs_cond_first_{start_idx}_{end_idx}.pt')
    torch.save(evecs_cond_second_tensor, f'{train_folder}/evecs_cond_second_{start_idx}_{end_idx}.pt')
    

    if pair_type == 'pair':
        torch.save(evecs_first_with_corr_tensor, f'{train_folder}/evecs_first_with_corr_{start_idx}_{end_idx}.pt')
        torch.save(evecs_second_with_corr_tensor, f'{train_folder}/evecs_second_with_corr_{start_idx}_{end_idx}.pt')
    


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_workers', type=int)
    parser.add_argument('--current_worker', type=int)
    
    parser.add_argument('--num_evecs', type=int)
    
    parser.add_argument('--net_path', type=str)
    # parser.add_argument('--net_input_type', type=str)
    # parser.add_argument('--evecs_per_support', type=int)
    
    parser.add_argument('--dataset_name', type=str)
    
    parser.add_argument('--pair_type', choices=['template', 'pair'])
    parser.add_argument('--n_pairs', type=int, required=False, default=1)
    
    parser.add_argument('--template_type', type=str)
    
    parser.add_argument('--regularization_lambda', type=float, required=False)
    
    parser.add_argument('--partial', type=float, required=True)
    
    
    args = parser.parse_args()
    
    # python my_code/datasets/cache_surreal_sign_corr.py --n_workers 20000 --current_worker 0 --num_evecs 32 --net_path /home/s94zalek_hpc/shape_matching/my_code/experiments/sign_net/test_partial_0.8_5k_32_1 --dataset_name test_partial_lambda_0.1 --template_type remeshed --pair_type template --n_pairs 1 --regularization_lambda 0.1
    
    return args
         
         
if __name__ == '__main__':
    
    args = parse_args()

    np.random.seed(120)
    
    num_evecs = args.num_evecs
        
    ####################################################
    # Dataset
    ####################################################
    
    # load the config
    with open(f'{args.net_path}/config.yaml', 'r') as f:
        sign_net_config = yaml.load(f, Loader=yaml.FullLoader)
           
    
    
    if args.partial > 0:
        
        # print('!!!!!!! for partial, no full meshes are included')
    
        augmentations = {
            "remesh": {
                "isotropic": {
                    "n_remesh_iters": 10,
                    "remesh_targetlen": 1,
                    "simplify_strength_min": 0.2,
                    "simplify_strength_max": 0.8,
                },
                "anisotropic": {
                    "probability": 0.35,
                        
                    "n_remesh_iters": 10,
                    "fraction_to_simplify_min": 0.2,
                    "fraction_to_simplify_max": 0.6,
                    "simplify_strength_min": 0.2,
                    "simplify_strength_max": 0.5,
                    "weighted_by": "face_count",
                },
                "partial": {
                    "probability": args.partial,
                    "n_remesh_iters": 10,
                    "fraction_to_keep_min": 0.4,
                    "fraction_to_keep_max": 0.8,
                    "n_seed_samples": [5, 25],
                    "weighted_by": "face_count",
                },
            },
        }
    else:
        # full
        
        augmentations = {
            "remesh": {
                "isotropic": {
                    "n_remesh_iters": 10,
                    "remesh_targetlen": 1,
                    "simplify_strength_min": 0.2,
                    "simplify_strength_max": 0.8,
                },
                "anisotropic": {
                    "probability": 0.35,
                        
                    "n_remesh_iters": 10,
                    "fraction_to_simplify_min": 0.2,
                    "fraction_to_simplify_max": 0.6,
                    "simplify_strength_min": 0.2,
                    "simplify_strength_max": 0.5,
                    "weighted_by": "face_count",
                },
            },
        }
    
     
    
    dataset = TemplateSurrealDataset3DC(
        shape_path='/lustre/mlnvme/data/s94zalek_hpc-shape_matching/mmap_datas_surreal_train.pth',
        num_evecs=128,
        cache_lb_dir=None,
        return_evecs=True,
        return_fmap=False,
        mmap=True,
        augmentations=augmentations,
        template_path=f'/home/s94zalek_hpc/shape_matching/data/SURREAL_full/template/{args.template_type}/template.off',
        template_corr=np.loadtxt(
            f'/home/s94zalek_hpc/shape_matching/data/SURREAL_full/template/{args.template_type}/corr.txt',
            dtype=np.int32) - 1,
        centering='mean',
        return_shot=sign_net_config['net_params']['input_type'] == 'shot',
    )   
    
    print('Dataset created')
    
    # sample train/test indices
    train_indices = list(range(len(dataset)))
    print(f'Number of training samples: {len(train_indices)}')
    
    # folder to store the dataset
    dataset_name = args.dataset_name
    # dataset_folder = f'/home/{user_name}/shape_matching/data/SURREAL_full/full_datasets/{dataset_name}'
    dataset_folder = f'/lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL/train/{dataset_name}'
    os.makedirs(dataset_folder, exist_ok=True)
    
    
    ####################################################
    # Sign correction network
    ####################################################

    # initialize the network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = diffusion_network.DiffusionNet(
        **sign_net_config['net_params']
        ).to(device)
    net.load_state_dict(torch.load(
        f'{args.net_path}/{sign_net_config["n_iter"]}.pth',
        map_location=device))
    
    
    ####################################################  
    # update the config
    ####################################################
    
    sign_net_config['net_path'] = args.net_path
    sign_net_config['augmentations'] = augmentations
    sign_net_config['template_type'] = args.template_type
    sign_net_config['pair_type'] = args.pair_type
    sign_net_config['n_pairs'] = args.n_pairs
    sign_net_config['regularization_lambda'] = args.regularization_lambda
    
    if args.pair_type == 'template':
        assert args.n_pairs == 1, 'n_pairs must be 1 for template pair type'
    
    # save the config to the dataset folder
    if not os.path.exists(f'{dataset_folder}/config.yaml'):
        with open(f'{dataset_folder}/config.yaml', 'w') as f:
            yaml.dump(sign_net_config, f)
            
            
    ####################################################
    # Regularization
    ####################################################
    
    if args.regularization_lambda is not None and args.regularization_lambda > 0:
        fmnet = RegularizedFMNet(
            lmbda=args.regularization_lambda,
            resolvant_gamma=0.5).to(device)
    else:
        fmnet = None
    
    ####################################################
    # Saving
    ####################################################

    # current and total workers
    n_workers = args.n_workers
    current_worker = args.current_worker
    
    # samples per worker
    n_samples = len(train_indices)
    samples_per_worker = n_samples // n_workers
    
    # start - end indices
    start = current_worker * samples_per_worker
    end = (current_worker + 1) * samples_per_worker
    if current_worker == n_workers - 1:
        end = n_samples
        
    print(f'Worker {current_worker} processing samples from {start} to {end}')
    
    # indices for this worker
    train_indices = train_indices[start:end]
    
    # subset = torch.utils.data.Subset(dataset, train_indices)
        

    print(f"Saving train dataset...")
    save_train_dataset(
        dataset=dataset,
        train_indices=train_indices,
        dataset_folder=dataset_folder,
        start_idx=start,
        end_idx=end,
        num_evecs=num_evecs,
        pair_type=args.pair_type,
        n_pairs=args.n_pairs,
        
        # sign corr net parameters
        net=net,
        net_input_type=sign_net_config['net_params']['input_type'],
        with_mass=sign_net_config['with_mass'],
        evecs_per_support=sign_net_config['evecs_per_support'],
        
        fmnet=fmnet
    )
