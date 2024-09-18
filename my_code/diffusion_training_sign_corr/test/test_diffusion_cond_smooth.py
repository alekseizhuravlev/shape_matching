import sqlite3
import torch
import numpy as np
import os
import shutil
from tqdm import tqdm
import yaml

import sys
import os

# models
from my_code.models.diag_conditional import DiagConditionedUnet
from diffusers import DDPMScheduler

import my_code.datasets.template_dataset as template_dataset

import my_code.diffusion_training_sign_corr.data_loading as data_loading

import networks.diffusion_network as diffusion_network
import matplotlib.pyplot as plt
import my_code.utils.plotting_utils as plotting_utils
import utils.fmap_util as fmap_util
import metrics.geodist_metric as geodist_metric
from my_code.sign_canonicalization.training import predict_sign_change
import argparse
from pyFM_fork.pyFM.refine.zoomout import zoomout_refine
import my_code.utils.zoomout_custom as zoomout_custom
import my_code.sign_canonicalization.test_sign_correction as test_sign_correction

import accelerate

from utils.shape_util import compute_geodesic_distmat
from my_code.utils.median_p2p_map import get_median_p2p_map, dirichlet_energy



tqdm._instances.clear()


def parse_args():
    parser = argparse.ArgumentParser(description='Train the model')
    
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--checkpoint_name', type=str)
    
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    
    parser.add_argument('--smoothing_type', choices=['laplacian', 'taubin'])
    parser.add_argument('--smoothing_iter', type=int)
    
    parser.add_argument('--num_iters_avg', type=int)
    
    args = parser.parse_args()
    return args

# python /home/s94zalek_hpc/shape_matching/my_code/diffusion_training_sign_corr/test/test_diffusion_cond_smooth.py --experiment_name=pair_5_xy_distributed --checkpoint_name=epoch_99 --dataset_name=FAUST_r_pair --split=test --smoothing_type=taubin --smoothing_iter=5


if __name__ == '__main__':

    args = parse_args()

    # configuration
    experiment_name = args.experiment_name
    checkpoint_name = args.checkpoint_name

    ### config
    exp_base_folder = f'/home/s94zalek_hpc/shape_matching/my_code/experiments/ddpm/{experiment_name}'
    with open(f'{exp_base_folder}/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    ### model
    model = DiagConditionedUnet(config["model_params"]).to('cuda')
    # model.load_state_dict(torch.load(f"{exp_base_folder}/checkpoints/{checkpoint_name}"))
    
    if "accelerate" in config and config["accelerate"]:
        accelerate.load_checkpoint_in_model(model, f"{exp_base_folder}/checkpoints/{checkpoint_name}/model.safetensors")
    else:
        model.load_state_dict(torch.load(f"{exp_base_folder}/checkpoints/{checkpoint_name}"))
    
    model = model.to('cuda')
    
    
    # algorithm
    # smooth the single dataset
    # for each mesh, correct the first evecs, get the conditioning
    
    # for each pair
    # sample the model with conditioning
    # zoomout using corrected evecs
    
    
    
    ### Sign correction network
    sign_corr_net = diffusion_network.DiffusionNet(
        **config["sign_net"]["net_params"]
        ).to('cuda')
        
    sign_corr_net.load_state_dict(torch.load(
            f'{config["sign_net"]["net_path"]}/{config["sign_net"]["n_iter"]}.pth'
            ))

    ### sample the model
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                                    clip_sample=True)

    ### test dataset
    dataset_name = args.dataset_name
    split = args.split

    single_dataset, test_dataset = data_loading.get_val_dataset(
        dataset_name, split, 200, preload=False, return_evecs=True
        )
    # sign_corr_net.cache_dir = single_dataset.lb_cache_dir

    single_dataset_remeshed = test_sign_correction.remesh_dataset(
        dataset=single_dataset, 
        name=dataset_name,
        remesh_targetlen=1,
        smoothing_type=args.smoothing_type,
        smoothing_iter=args.smoothing_iter,
        num_evecs=200,
    )


    num_evecs = config["model_params"]["sample_size"]

    ##########################################
    # Logging
    ##########################################

    log_dir = f'{exp_base_folder}/eval/{checkpoint_name}/{dataset_name}-{split}/{args.smoothing_type}-{args.smoothing_iter}'
    os.makedirs(log_dir, exist_ok=True)

    fig_dir = f'{log_dir}/figs'
    os.makedirs(fig_dir, exist_ok=True)

    log_file_name = f'{log_dir}/log_smooth_{args.smoothing_type}_{args.smoothing_iter}.txt'

    ##########################################
    # Single stage
    ##########################################

    data_range = tqdm(range(len(single_dataset_remeshed)), desc='Calculating conditioning, correcting evecs')

    for i in data_range:

        data = single_dataset_remeshed[i]
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        verts_second = data['verts'].unsqueeze(0).to(device)
        faces_second = data['faces'].unsqueeze(0).to(device)
        
        evecs_second = data['evecs'][:, :num_evecs].unsqueeze(0).to(device)
        evals_second = data['evals'][:num_evecs]

        # corr_second = data['corr']
        
        if config["sign_net"]["with_mass"]:
            mass_mat_second = torch.diag_embed(
                data['mass'].unsqueeze(0)
                ).to(device)
        else:
            mass_mat_second = None

        # predict the sign change
        with torch.no_grad():
            sign_pred_second, support_vector_norm_second, _ = predict_sign_change(
                sign_corr_net, verts_second, faces_second, evecs_second, 
                mass_mat=mass_mat_second, input_type=sign_corr_net.input_type,
                # mass=None, L=None, evals=None, evecs=None, gradX=None, gradY=None
                mass=data['mass'].unsqueeze(0), L=data['L'].unsqueeze(0),
                evals=data['evals'][:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                evecs=data['evecs'][:,:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                gradX=data['gradX'].unsqueeze(0), gradY=data['gradY'].unsqueeze(0)
                )

        # correct the evecs
        evecs_second_corrected = evecs_second.cpu()[0] * torch.sign(sign_pred_second).cpu()
        evecs_second_corrected_norm = evecs_second_corrected / torch.norm(evecs_second_corrected, dim=0, keepdim=True)
        
        # product with support
        if config["sign_net"]["with_mass"]:
            mass_mat_second = torch.diag_embed(
                data['mass'].unsqueeze(0)
                ).to(device)
            
            evecs_cond_second = torch.nn.functional.normalize(
                support_vector_norm_second[0].cpu().transpose(0, 1) \
                    @ mass_mat_second[0].cpu(),
                p=2, dim=1) \
                    @ evecs_second_corrected_norm       
        else:
            evecs_cond_second = support_vector_norm_second[0].cpu().transpose(0, 1) @ evecs_second_corrected_norm
        
        
        ###############################################
        # Conditioning
        ###############################################

        conditioning = torch.tensor([])
        
        if 'evals' in config["conditioning_types"]:
            eval = evals_second.unsqueeze(0)
            eval = torch.diag_embed(eval)
            conditioning = torch.cat((conditioning, eval), 0)
        
        if 'evals_inv' in config["conditioning_types"]:
            eval_inv = 1 / evals_second.unsqueeze(0)
            # replace elements > 1 with 1
            eval_inv[eval_inv > 1] = 1
            eval_inv = torch.diag_embed(eval_inv)
            conditioning = torch.cat((conditioning, eval_inv), 0)
        
        if 'evecs' in config["conditioning_types"]:
            conditioning = torch.cat((conditioning,
                                      evecs_cond_second.unsqueeze(0)), 0)
        
        ###############################################
        # Correct the original evecs
        ###############################################
        
        data_orig = single_dataset[i]
        evecs_second_orig = data_orig['evecs'][:, :num_evecs]
        
        prod_evecs_orig_remesh_corrected = evecs_second_orig.transpose(0, 1) @ evecs_second_corrected[data['corr_orig_to_remeshed']].cpu()

        evecs_orig_signs = torch.sign(torch.diagonal(prod_evecs_orig_remesh_corrected, dim1=0, dim2=1))
        evecs_second_corrected_orig = evecs_second_orig * evecs_orig_signs
        
        evecs_second_orig_zo = torch.cat(
            [evecs_second_corrected_orig,
                data_orig['evecs'][:, num_evecs:]], 1)

        ###############################################
        # Save the data
        ###############################################

        single_dataset.additional_data[i]['evecs_zo'] = evecs_second_orig_zo
        single_dataset.additional_data[i]['conditioning'] = conditioning
        

    ##########################################
    # Pairwise stage
    ##########################################
        
    test_dataset.dataset = single_dataset
        
    ratios = []
    geo_errs = []
    geo_errs_pairzo = []
    
    geo_errs_median = []
    geo_errs_pairzo_median = []
    geo_errs_pairzo_median_p2p = []
    geo_errs_pairzo_dirichlet = []

    Cxy_est_list = []
    C_gt_xy_corr_list = []

        
    data_range_pair = tqdm(range(len(test_dataset)), desc='Calculating pair fmaps')

    # data_range_pair = tqdm(range(2))
    # print('!!!!! WARNING: only 5 meshes are processed !!!!!')

    for i in data_range_pair:
        
        data = test_dataset[i]
        
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'
        
        verts_first = data['first']['verts'].to(device)
        verts_second = data['second']['verts'].to(device)
        
        faces_first = data['first']['faces'].to(device)
        faces_second = data['second']['faces'].to(device)

        # evecs_first = data['first']['evecs'][:, :num_evecs].unsqueeze(0).to(device)
        # evecs_second = data['second']['evecs'][:, :num_evecs].unsqueeze(0).to(device)
        
        evecs_first = data['first']['evecs'][:, :].to(device)
        evecs_second = data['second']['evecs'][:, :].to(device)
        
        evals_first = data['first']['evals'][:num_evecs]
        evals_second = data['second']['evals'][:num_evecs]

        corr_first = data['first']['corr'].to(device)
        corr_second = data['second']['corr'].to(device)
        
        ###############################################
        # Functional maps
        ###############################################
        
        evecs_first_zo = data['first']['evecs_zo'].to(device)
        evecs_second_zo = data['second']['evecs_zo'].to(device)
        
        conditioning_first = data['first']['conditioning'].to(device)
        conditioning_second = data['second']['conditioning'].to(device)
        
        ###############################################
        # Sample the model
        ###############################################
        
        # x_sampled = torch.rand(1, 1, model.model.sample_size, model.model.sample_size).to(device)
        # y = torch.cat(
        #     (conditioning_first, conditioning_second),
        #               0).unsqueeze(0).to(device)    
        
        x_sampled = torch.rand(args.num_iters_avg, 1, model.model.sample_size, model.model.sample_size).to(device)
        y = torch.cat(
            (conditioning_first, conditioning_second),
                    0).unsqueeze(0).repeat(args.num_iters_avg, 1, 1, 1).to(device)
         
        # Sampling loop
        for t in noise_scheduler.timesteps:

            # Get model pred
            with torch.no_grad():
                residual = model(x_sampled, t,
                                    conditioning=y
                                    ).sample

            # Update sample with step
            x_sampled = noise_scheduler.step(residual, t, x_sampled).prev_sample

        ###############################################
        
        dist_x = torch.tensor(
            compute_geodesic_distmat(data['first']['verts'].numpy(), data['first']['faces'].numpy())    
        )

        geo_err_est_sampled = []
        geo_err_est_pairzo_sampled = []
        p2p_est_pairzo_sampled = []
        
        for k in range(args.num_iters_avg):

            Cxy_est_k = x_sampled[k][0]
            
            ###############################################
            
            C_gt_xy = torch.linalg.lstsq(
                evecs_second[corr_second],
                evecs_first[corr_first]
                ).solution
            
            C_gt_xy_corr = torch.linalg.lstsq(
                evecs_second_zo[corr_second],
                evecs_first_zo[corr_first]
                ).solution
            
            Cxy_est_pairzo_k = zoomout_custom.zoomout(
                FM_12=Cxy_est_k, 
                evects1=evecs_first_zo, 
                evects2=evecs_second_zo,
                nit=evecs_first_zo.shape[1] - num_evecs, step=1,
            )
            
            ###############################################
            # Evaluation
            ###############################################  
            
            # hard correspondence 
            p2p_gt = fmap_util.fmap2pointmap(
                C12=C_gt_xy,
                evecs_x=evecs_first,
                evecs_y=evecs_second,
                ).cpu()
            p2p_corr_gt = fmap_util.fmap2pointmap(
                C12=C_gt_xy_corr,
                evecs_x=evecs_first_zo,
                evecs_y=evecs_second_zo,
                ).cpu()
            p2p_est_k = fmap_util.fmap2pointmap(
                Cxy_est_k,
                evecs_x=evecs_first_zo[:, :num_evecs],
                evecs_y=evecs_second_zo[:, :num_evecs],
                ).cpu()
            p2p_est_pairzo_k = fmap_util.fmap2pointmap(
                Cxy_est_pairzo_k,
                evecs_x=evecs_first_zo,
                evecs_y=evecs_second_zo,
                ).cpu()
            
            # distance matrices
            # dist_x = torch.cdist(data['first']['verts'], data['first']['verts'])
            # dist_y = torch.cdist(data['second']['verts'], data['second']['verts'])
            
                
            # geodesic error
            geo_err_gt = geodist_metric.calculate_geodesic_error(
                dist_x, data['first']['corr'], data['second']['corr'], p2p_gt, return_mean=False
                )  
            geo_err_corr_gt = geodist_metric.calculate_geodesic_error(
                dist_x, data['first']['corr'], data['second']['corr'], p2p_corr_gt, return_mean=False
                )
            geo_err_est_k = geodist_metric.calculate_geodesic_error(
                dist_x, data['first']['corr'], data['second']['corr'], p2p_est_k, return_mean=False
                )
            geo_err_est_pairzo_k = geodist_metric.calculate_geodesic_error(
                dist_x, data['first']['corr'], data['second']['corr'], p2p_est_pairzo_k, return_mean=False
                )
            
            geo_err_est_sampled.append(geo_err_est_k.mean())
            geo_err_est_pairzo_sampled.append(geo_err_est_pairzo_k.mean())
            p2p_est_pairzo_sampled.append(p2p_est_pairzo_k)
            

        geo_err_est_sampled = torch.tensor(geo_err_est_sampled)
        geo_err_est_pairzo_sampled = torch.tensor(geo_err_est_pairzo_sampled)
        
        ###############################################
        # p2p map selection
        ###############################################
        
        # median p2p map
        p2p_est_pairzo_sampled = torch.stack(p2p_est_pairzo_sampled)
        
        # dirichlet energy for each p2p map
        dirichlet_energy_list = []
        for n in range(p2p_est_pairzo_sampled.shape[0]):
            dirichlet_energy_list.append(
                dirichlet_energy(p2p_est_pairzo_sampled[n], verts_first.cpu(), data['second']['L']).item(),
                )
        dirichlet_energy_list = torch.tensor(dirichlet_energy_list)

        # sort by dirichlet energy, get the arguments
        _, sorted_idx_dirichlet = torch.sort(dirichlet_energy_list)
        
        # map with the lowest dirichlet energy
        p2p_dirichlet = p2p_est_pairzo_sampled[sorted_idx_dirichlet[0]]
        
        # median p2p map, using 3 maps with lowest dirichlet energy
        p2p_median = get_median_p2p_map(
            p2p_est_pairzo_sampled[
                sorted_idx_dirichlet[:int(round(args.num_iters_avg / 10))]
                ],
            dist_x
            )
        
        geo_err_est_pairzo_median = geodist_metric.calculate_geodesic_error(
            dist_x, data['first']['corr'], data['second']['corr'], p2p_median, return_mean=True
                )
        geo_err_est_pairzo_dirichlet = geodist_metric.calculate_geodesic_error(
            dist_x, data['first']['corr'], data['second']['corr'], p2p_dirichlet, return_mean=True
                )
        
        sorted_idxs_geo_err_zo = torch.argsort(geo_err_est_pairzo_sampled)

        # replace code above with writing to log file
        with open(log_file_name, 'a') as f:
            f.write(f'{i}\n')
            f.write(f'Geo error GT: {geo_err_gt.mean() * 100:.2f}\n')
            f.write(f'Geo error GT corr: {geo_err_corr_gt.mean() * 100:.2f}\n')
            # f.write(f'Geo error est: {geo_err_est_sampled.mean() * 100:.2f}\n')
            # f.write(f'Geo error est pairzo: {geo_err_est_pairzo_sampled.mean() * 100:.2f}\n')
            # f.write('-----------------------------------\n')
            
            f.write(f'Geo error est mean: {geo_err_est_sampled.mean() * 100:.2f}, \n'+\
            f'Geo error est median: {geo_err_est_sampled.median() * 100:.2f}, \n'+\
            f'Geo error est: {geo_err_est_sampled[sorted_idxs_geo_err_zo] * 100}, \n'
            f'Geo error est zo mean: {geo_err_est_pairzo_sampled.mean() * 100:.2f}, \n'+\
            f'Geo error est zo median: {geo_err_est_pairzo_sampled.median() * 100:.2f}\n'
            f'Geo error est zo: {geo_err_est_pairzo_sampled[sorted_idxs_geo_err_zo] * 100}\n'
            f'Geo error est p2p zo median: {geo_err_est_pairzo_median * 100:.2f}\n'
            f'Geo error est p2p zo dirichlet: {geo_err_est_pairzo_dirichlet * 100:.2f}\n'
            f'Dirichlet energy: {dirichlet_energy_list[sorted_idxs_geo_err_zo]}\n'
            )
            f.write('-----------------------------------\n')
        
        
        
        ratio_curr = geo_err_est_sampled.mean() / geo_err_corr_gt.mean()
        
        ratios.append(ratio_curr)
        geo_errs.append(geo_err_est_sampled.mean() * 100)
        geo_errs_pairzo.append(geo_err_est_pairzo_sampled.mean() * 100)
        
        geo_errs_median.append(geo_err_est_sampled.median() * 100)
        geo_errs_pairzo_median.append(geo_err_est_pairzo_sampled.median() * 100)
        geo_errs_pairzo_median_p2p.append(geo_err_est_pairzo_median * 100)
        geo_errs_pairzo_dirichlet.append(geo_err_est_pairzo_dirichlet * 100)
        
        # if i % 10 == 0:
        data_range_pair.set_description(
            f'Mean {torch.tensor(geo_errs).mean():.2f}, '+\
            f'pairzo {torch.tensor(geo_errs_pairzo).mean():.2f}, '
            )


    ratios = torch.tensor(ratios)
    geo_errs = torch.tensor(geo_errs)
    geo_errs_pairzo = torch.tensor(geo_errs_pairzo)
    
    geo_errs_median = torch.tensor(geo_errs_median)
    geo_errs_pairzo_median = torch.tensor(geo_errs_pairzo_median)
    geo_errs_pairzo_median_p2p = torch.tensor(geo_errs_pairzo_median_p2p)
    geo_errs_pairzo_dirichlet = torch.tensor(geo_errs_pairzo_dirichlet)
        
    # replace code above with writing to log file
    with open(log_file_name, 'a') as f:
        f.write('-----------------------------------\n')
        f.write('Total statistics\n')
        f.write(f'Pairzoomout geo err mean: {geo_errs_pairzo.mean():.2f}\n')
        f.write(f'Pairzoomout geo err median: {geo_errs_pairzo.median():.2f}\n')
        f.write(f'Pairzoomout geo err min: {geo_errs_pairzo.min():.2f}\n')
        f.write(f'Pairzoomout geo err max: {geo_errs_pairzo.max():.2f}\n')      
        f.write('-----------------------------------\n')
        f.write(f'Mean geo err: {geo_errs.mean():.2f}\n')
        f.write(f'Median geo err: {geo_errs.median():.2f}\n')
        f.write(f'Min geo err: {geo_errs.min():.2f}\n')
        f.write(f'Max geo err: {geo_errs.max():.2f}\n')
        f.write('-----------------------------------\n')
        f.write(f'Medians\n')
        f.write(f'Mean geo err: {geo_errs_median.mean():.2f}\n')
        f.write(f'Pairzoomout geo err mean: {geo_errs_pairzo_median.mean():.2f}\n')
        f.write('\n')
        f.write(f'geo_errs_pairzo_median_p2p: {geo_errs_pairzo_median_p2p.mean():.2f}\n')
        f.write(f'geo_errs_pairzo_dirichlet: {geo_errs_pairzo_dirichlet.mean():.2f}\n')
        f.write('-----------------------------------\n')
        
        
    
    # log to database    
    con = sqlite3.connect("/home/s94zalek_hpc/shape_matching/my_code/experiments/ddpm/log_p2p_median_dirichlet.db")
    cur = con.cursor()
    
    data = [(
        args.experiment_name,
        args.checkpoint_name, 
        f'{args.smoothing_type}-{args.smoothing_iter}', 
        args.dataset_name,
        args.split, 
        # dirichlet
        geo_errs_pairzo_dirichlet.mean().item(),
        # p2p median
        geo_errs_pairzo_median_p2p.mean().item(),
        # zoomout
        geo_errs_pairzo.mean().item(), geo_errs_pairzo_median.median().item(),
        # pred
        geo_errs.mean().item(), geo_errs_median.median().item()
        ),]
    
    # if an entry with the same first 5 entries exists, delete it
    
    if cur.execute(f"SELECT * FROM ddpm WHERE experiment_name='{args.experiment_name}' AND checkpoint_name='{args.checkpoint_name}' AND smoothing='{args.smoothing_type}-{args.smoothing_iter}' AND dataset_name='{args.dataset_name}' AND split='{args.split}'").fetchone():
        print('Deleting existing entry')
        cur.execute(f"DELETE FROM ddpm WHERE experiment_name='{args.experiment_name}' AND checkpoint_name='{args.checkpoint_name}' AND smoothing='{args.smoothing_type}-{args.smoothing_iter}' AND dataset_name='{args.dataset_name}' AND split='{args.split}'")
        
    
    cur.executemany("INSERT INTO ddpm VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data)
    con.commit()
    
    con.close()