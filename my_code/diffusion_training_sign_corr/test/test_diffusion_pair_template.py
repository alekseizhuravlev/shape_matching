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
from utils.shape_util import compute_geodesic_distmat
from my_code.diffusion_training_sign_corr.test.test_diffusion_cond import select_p2p_map_dirichlet, log_to_database, parse_args
        

tqdm._instances.clear()


def get_geo_error(
    p2p_first, p2p_second,
    evecs_first, evecs_second,
    corr_first, corr_second,
    num_evecs, apply_zoomout,
    dist_x
    ):
    Cxy = torch.linalg.lstsq(
        evecs_second[:, :num_evecs][p2p_second],
        evecs_first[:, :num_evecs][p2p_first]
        ).solution
    
    if apply_zoomout:
        Cxy = zoomout_custom.zoomout(
            FM_12=Cxy, 
            evects1=evecs_first,
            evects2=evecs_second,
            nit=evecs_first.shape[1] - num_evecs, step=1,
        )
        num_evecs = evecs_first.shape[1]
        
    p2p = fmap_util.fmap2pointmap(
        C12=Cxy,
        evecs_x=evecs_first[:, :num_evecs],
        evecs_y=evecs_second[:, :num_evecs],
        ).cpu()
    
    geo_err = geodist_metric.calculate_geodesic_error(
        dist_x, corr_first.cpu(), corr_second.cpu(), p2p, return_mean=True
    )
    
    return geo_err * 100


def run():

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
    model.load_state_dict(torch.load(f"{exp_base_folder}/checkpoints/{checkpoint_name}"))
    model = model.to('cuda')
    
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
    sign_corr_net.cache_dir = single_dataset.lb_cache_dir


    num_evecs = config["model_params"]["sample_size"]


    ##########################################
    # Template
    ##########################################

    template_shape = template_dataset.get_template(
        # template_path='data/SURREAL_full/template/template.ply',
        num_evecs=single_dataset.num_evecs,
        # template_corr=list(range(6890)),
        centering='bbox',
        
        template_path=f'/home/s94zalek_hpc/shape_matching/data/SURREAL_full/template/{config["sign_net"]["template_type"]}/template.off',
        template_corr=np.loadtxt(
            f'/home/s94zalek_hpc/shape_matching/data/SURREAL_full/template/{config["sign_net"]["template_type"]}/corr.txt',
            dtype=np.int32) - 1
        )    

    ##########################################
    # Logging
    ##########################################

    # log_dir = f'{exp_base_folder}/eval/{checkpoint_name}/{dataset_name}-{split}-template'
    # os.makedirs(log_dir, exist_ok=True)

    # fig_dir = f'{log_dir}/figs'
    # os.makedirs(fig_dir, exist_ok=True)

    # log_file_name = f'{log_dir}/log.txt'
    
    log_dir = f'{exp_base_folder}/eval/{checkpoint_name}/{dataset_name}-{split}/no_smoothing'
    os.makedirs(log_dir, exist_ok=True)

    fig_dir = f'{log_dir}/figs'
    os.makedirs(fig_dir, exist_ok=True)

    log_file_name = f'{log_dir}/log.txt'


    ##########################################
    # Template stage
    ##########################################

    data_range = tqdm(range(len(single_dataset)), desc='Calculating fmaps to template')
    
    # data_range = tqdm(range(2))
    # print('!!! WARNING: only 2 samples are processed !!!')

    for i in data_range:

        data = single_dataset[i]
        
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'
        
        verts_first = template_shape['verts'].unsqueeze(0).to(device)
        verts_second = data['verts'].unsqueeze(0).to(device)
        
        faces_first = template_shape['faces'].unsqueeze(0).to(device)
        faces_second = data['faces'].unsqueeze(0).to(device)

        evecs_first = template_shape['evecs'][:, :num_evecs].unsqueeze(0).to(device)
        evecs_second = data['evecs'][:, :num_evecs].unsqueeze(0).to(device)
        
        evals_first = template_shape['evals'][:num_evecs]
        evals_second = data['evals'][:num_evecs]

        # corr_first = data['first']['corr']
        # corr_second = data['corr']
        
        if config["sign_net"]["with_mass"]:
            mass_mat_first = torch.diag_embed(
                template_shape['mass'].unsqueeze(0)
                ).to(device)
            mass_mat_second = torch.diag_embed(
                data['mass'].unsqueeze(0)
                ).to(device)
        else:
            mass_mat_first = None
            mass_mat_second = None


        # predict the sign change
        with torch.no_grad():
            sign_pred_first, support_vector_norm_first, _ = predict_sign_change(
                sign_corr_net, verts_first, faces_first, evecs_first, 
                mass_mat=mass_mat_first, input_type=sign_corr_net.input_type,
                # mass=None, L=None, evals=None, evecs=None, gradX=None, gradY=None
                mass=template_shape['mass'].unsqueeze(0), L=template_shape['L'].unsqueeze(0),
                evals=template_shape['evals'][:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                evecs=template_shape['evecs'][:,:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                gradX=template_shape['gradX'].unsqueeze(0), gradY=template_shape['gradY'].unsqueeze(0)
                )
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
        evecs_first_corrected = evecs_first.cpu()[0] * torch.sign(sign_pred_first).cpu()
        evecs_first_corrected_norm = evecs_first_corrected / torch.norm(evecs_first_corrected, dim=0, keepdim=True)
        
        evecs_second_corrected = evecs_second.cpu()[0] * torch.sign(sign_pred_second).cpu()
        evecs_second_corrected_norm = evecs_second_corrected / torch.norm(evecs_second_corrected, dim=0, keepdim=True)
        
        # product with support
        # evecs_cond_first = evecs_first_corrected_norm.transpose(0, 1) @ support_vector_norm_first[0].cpu()
        # evecs_cond_second = evecs_second_corrected_norm.transpose(0, 1) @ support_vector_norm_second[0].cpu()


        # product with support
        if config["sign_net"]["with_mass"]:
        # if config["sign_net"]['cond_mass_normalize']:
            
            mass_mat_first = torch.diag_embed(
                template_shape['mass'].unsqueeze(0)
                ).to(device)
            mass_mat_second = torch.diag_embed(
                data['mass'].unsqueeze(0)
                ).to(device)
            
            evecs_cond_first = torch.nn.functional.normalize(
                support_vector_norm_first[0].cpu().transpose(0, 1) \
                    @ mass_mat_first[0].cpu(),
                p=2, dim=1) \
                    @ evecs_first_corrected_norm
            
            evecs_cond_second = torch.nn.functional.normalize(
                support_vector_norm_second[0].cpu().transpose(0, 1) \
                    @ mass_mat_second[0].cpu(),
                p=2, dim=1) \
                    @ evecs_second_corrected_norm 
            
        else:
            evecs_cond_first = support_vector_norm_first[0].cpu().transpose(0, 1) @ evecs_first_corrected_norm
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
            evecs = torch.cat(
                (evecs_cond_first.unsqueeze(0), evecs_cond_second.unsqueeze(0)),
                0)
            conditioning = torch.cat((conditioning, evecs), 0)
        
        
        ###############################################
        # Sample the model
        ###############################################
        
        x_sampled = torch.rand(args.num_iters_avg, 1, model.model.sample_size, model.model.sample_size).to(device)
        y = conditioning.unsqueeze(0).repeat(args.num_iters_avg, 1, 1, 1).to(device)    
        
        # print(x_sampled.shape, y.shape)
            
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
        # Zoomout
        ###############################################
        
        evecs_first_zo = torch.cat(
            [evecs_first_corrected,
                template_shape['evecs'][:, num_evecs:]], 1)
        evecs_second_zo = torch.cat(
            [evecs_second_corrected,
                data['evecs'][:, num_evecs:]], 1)
        
        
        # single_dataset.additional_data[i]['Cyx_est'] = []
        # single_dataset.additional_data[i]['Cyx_est_zo'] = []
        single_dataset.additional_data[i]['evecs_zo'] = evecs_second_zo

        single_dataset.additional_data[i]['p2p_est'] = []
        # single_dataset.additional_data[i]['p2p_est_zo'] = []
        
        for k in range(args.num_iters_avg):
            Cyx_est_k = x_sampled[k][0].cpu()
        
            Cyx_est_zo_k = zoomout_custom.zoomout(
                FM_12=Cyx_est_k.to(device), 
                evects1=evecs_second_zo.to(device), 
                evects2=evecs_first_zo.to(device),
                nit=evecs_first_zo.shape[1] - num_evecs, step=1,
            ).cpu()

            p2p_est_k = fmap_util.fmap2pointmap(
                C12=Cyx_est_k.to(device),
                evecs_x=evecs_second_corrected.to(device),
                evecs_y=evecs_first_corrected.to(device),
                ).cpu()

            p2p_est_zo_k = fmap_util.fmap2pointmap(
                C12=Cyx_est_zo_k.to(device),
                evecs_x=evecs_second_zo.to(device),
                evecs_y=evecs_first_zo.to(device),
                ).cpu()

            # single_dataset.additional_data[i]['Cyx_est'].append(Cyx_est_k)
            # single_dataset.additional_data[i]['Cyx_est_zo'].append(Cyx_est_zo_k)
            # single_dataset.additional_data[i]['evecs_zo'] = evecs_second_zo

            single_dataset.additional_data[i]['p2p_est'].append(p2p_est_k)
            # single_dataset.additional_data[i]['p2p_est_zo'].append(p2p_est_zo_k)
            
            
        single_dataset.additional_data[i]['p2p_est'] = torch.stack(single_dataset.additional_data[i]['p2p_est'])
            
        ##########################################################
        # p2p map selection
        ##########################################################
        
        dist_second = torch.tensor(
            compute_geodesic_distmat(
                verts_second[0].cpu().numpy(),
                faces_second[0].cpu().numpy())    
        )
        
        p2p_dirichlet, p2p_median, dirichlet_energy_list = select_p2p_map_dirichlet(
            single_dataset.additional_data[i]['p2p_est'],
            verts_second[0].cpu(),
            template_shape['L'], 
            dist_second
            )
        
        single_dataset.additional_data[i]['p2p_dirichlet'] = p2p_dirichlet
        single_dataset.additional_data[i]['p2p_median'] = p2p_median

    
    
    ##########################################
    # Pairwise stage
    ##########################################
        
    test_dataset.dataset = single_dataset
        
    geo_errs_gt = []
    geo_errs_corr_gt = []
    geo_errs_pairzo = []
    geo_errs_dirichlet = []
    geo_errs_median = []
    
        
    data_range_pair = tqdm(range(len(test_dataset)), desc='Calculating pair fmaps')

    # data_range_pair = tqdm(range(2))
    # print('!!! WARNING: only 2 samples are processed !!!')

    for i in data_range_pair:
        
        data = test_dataset[i]        
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'
        
        verts_first = data['first']['verts'].to(device)
        verts_second = data['second']['verts'].to(device)
        
        faces_first = data['first']['faces'].to(device)
        faces_second = data['second']['faces'].to(device)

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
        
        p2p_est_first = data['first']['p2p_est'].to(device)
        p2p_est_second = data['second']['p2p_est'].to(device)
        
        p2p_dirichlet_first = data['first']['p2p_dirichlet'].to(device)
        p2p_dirichlet_second = data['second']['p2p_dirichlet'].to(device)
        
        p2p_median_first = data['first']['p2p_median'].to(device)
        p2p_median_second = data['second']['p2p_median'].to(device)
        
        dist_x = torch.tensor(
            compute_geodesic_distmat(data['first']['verts'].numpy(), data['first']['faces'].numpy())    
        )
        
        ###############################################
        # Geodesic errors
        ###############################################
        
        # GT geo error
        geo_err_gt = get_geo_error(
            corr_first, corr_second,
            evecs_first, evecs_second,
            corr_first, corr_second,
            num_evecs, False,
            dist_x
            )
        geo_err_corr_gt = get_geo_error(
            corr_first, corr_second,
            evecs_first_zo, evecs_second_zo,
            corr_first, corr_second,
            num_evecs, False,
            dist_x
            )
        
        # mean pred geo error with zoomout
        geo_err_est_pairzo = []
        for k in range(args.num_iters_avg):
            geo_err_est_pairzo.append(
                get_geo_error(
                p2p_est_first[k], p2p_est_second[k],
                evecs_first_zo, evecs_second_zo,
                corr_first, corr_second,
                num_evecs, True,
                dist_x
                ))
        geo_err_est_pairzo = torch.tensor(geo_err_est_pairzo)
        
        # dirichlet geo error
        geo_err_est_dirichlet = get_geo_error(
            p2p_dirichlet_first, p2p_dirichlet_second,
            evecs_first_zo, evecs_second_zo,
            corr_first, corr_second,
            num_evecs, True,
            dist_x
            )
        
        # median geo error
        geo_err_est_median = get_geo_error(
            p2p_median_first, p2p_median_second,
            evecs_first_zo, evecs_second_zo,
            corr_first, corr_second,
            num_evecs, True,
            dist_x
            )

        # replace code above with writing to log file
        with open(log_file_name, 'a') as f:
            f.write(f'{i}\n')
            f.write(f'Geo error GT: {geo_err_gt:.2f}\n')
            f.write(f'Geo error GT corr: {geo_err_corr_gt:.2f}\n')
            f.write(f'Geo error est pairzo: {geo_err_est_pairzo}\n')
            f.write(f'Geo error est pairzo mean: {geo_err_est_pairzo.mean():.2f}\n')
            f.write(f'Geo error est dirichlet: {geo_err_est_dirichlet:.2f}\n')
            f.write(f'Geo error est median: {geo_err_est_median:.2f}\n')
            f.write('-----------------------------------\n')
        
        geo_errs_gt.append(geo_err_gt)
        geo_errs_corr_gt.append(geo_err_corr_gt)
        geo_errs_pairzo.append(geo_err_est_pairzo.mean())
        geo_errs_dirichlet.append(geo_err_est_dirichlet)
        geo_errs_median.append(geo_err_est_median)


    geo_errs_gt = torch.tensor(geo_errs_gt)
    geo_errs_corr_gt = torch.tensor(geo_errs_corr_gt)
    geo_errs_pairzo = torch.tensor(geo_errs_pairzo)
    geo_errs_dirichlet = torch.tensor(geo_errs_dirichlet)
    geo_errs_median = torch.tensor(geo_errs_median)
        
    # replace code above with writing to log file
    with open(log_file_name, 'a') as f:
        f.write('-----------------------------------\n')
        f.write('Total statistics\n')
        f.write('-----------------------------------\n')
        f.write(f'GT geo err mean: {geo_errs_gt.mean():.2f}\n')
        f.write(f'GT corr geo err mean: {geo_errs_corr_gt.mean():.2f}\n')
        f.write('\n')
        f.write(f'Pairzoomout geo err mean: {geo_errs_pairzo.mean():.2f}\n')
        f.write(f'Pairzoomout geo err median: {geo_errs_pairzo.median():.2f}\n')
        f.write(f'Pairzoomout geo err min: {geo_errs_pairzo.min():.2f}\n')
        f.write(f'Pairzoomout geo err max: {geo_errs_pairzo.max():.2f}\n')      
        f.write('\n')
        f.write(f'Dirichlet geo err mean: {geo_errs_dirichlet.mean():.2f}\n')
        f.write(f'Dirichlet geo err median: {geo_errs_dirichlet.median():.2f}\n')
        f.write(f'Dirichlet geo err min: {geo_errs_dirichlet.min():.2f}\n')
        f.write(f'Dirichlet geo err max: {geo_errs_dirichlet.max():.2f}\n')
        f.write('\n')
        f.write(f'Median geo err mean: {geo_errs_median.mean():.2f}\n')
        f.write(f'Median geo err median: {geo_errs_median.median():.2f}\n')
        f.write(f'Median geo err min: {geo_errs_median.min():.2f}\n')
        f.write(f'Median geo err max: {geo_errs_median.max():.2f}\n')
        f.write('-----------------------------------\n')
        
    data = [(
        args.experiment_name,
        args.checkpoint_name, 
        'no', 
        args.dataset_name,
        args.split, 
        # dirichlet
        geo_errs_dirichlet.mean().item(),
        # median p2p
        geo_errs_median.mean().item(),
        # zoomout
        geo_errs_pairzo.mean().item(), geo_errs_pairzo.median().item(),
        # pred
        0, 0
        ),]
    
    log_to_database(data)
        
        
if __name__ == '__main__':
    run()