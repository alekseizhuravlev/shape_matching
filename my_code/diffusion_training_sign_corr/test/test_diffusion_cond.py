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

import accelerate
import sqlite3

from utils.shape_util import compute_geodesic_distmat
from my_code.utils.median_p2p_map import get_median_p2p_map, dirichlet_energy

import datetime

tqdm._instances.clear()


def parse_args():
    parser = argparse.ArgumentParser(description='Test the model')
    
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--checkpoint_name', type=str, required=True)
    
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    
    parser.add_argument('--smoothing_type', choices=['laplacian', 'taubin'], required=False)
    parser.add_argument('--smoothing_iter', type=int, required=False)
    
    
    parser.add_argument('--num_iters_avg', type=int, required=True)
    parser.add_argument('--num_samples_median', type=int, required=True)
    parser.add_argument('--confidence_threshold', type=float, required=True)
    
    parser.add_argument('--log_subdir', type=str, required=True)
    
    parser.add_argument('--dirichlet_energy_threshold_template', type=float, required=False)
    parser.add_argument('--zoomout_num_evecs_template', type=int, required=False)
    
    parser.add_argument('--reduced', action='store_true', default=False)
    
    parser.add_argument('--random_seed', type=int, required=False)
    
    args = parser.parse_args()
    return args


def select_p2p_map_dirichlet(p2p_est_zo_sampled, verts_first, L_second, dist_first, num_samples_median):

    # dirichlet energy for each p2p map
    dirichlet_energy_list = []
    for n in range(p2p_est_zo_sampled.shape[0]):
        dirichlet_energy_list.append(
            dirichlet_energy(p2p_est_zo_sampled[n], verts_first, L_second).item(),
            )
    dirichlet_energy_list = torch.tensor(dirichlet_energy_list)

    # sort by dirichlet energy, get the arguments
    _, sorted_idx_dirichlet = torch.sort(dirichlet_energy_list)
    
    # map with the lowest dirichlet energy
    p2p_dirichlet = p2p_est_zo_sampled[sorted_idx_dirichlet[0]]
    
    # median p2p map, using 3 maps with lowest dirichlet energy
    p2p_median, confidence_scores = get_median_p2p_map(
        p2p_est_zo_sampled[
            sorted_idx_dirichlet[:num_samples_median]
            ],
        dist_first
        )
    
    return p2p_dirichlet, p2p_median, confidence_scores, dirichlet_energy_list


# def log_to_database(data):
    
#     assert len(data) == 1
#     assert len(data[0]) == 11
    
#     # log to database    
#     con = sqlite3.connect("/home/s94zalek_hpc/shape_matching/my_code/experiments/ddpm/log_p2p_median_dirichlet.db")
#     cur = con.cursor()
    
#     # if an entry with the same first 5 entries exists, delete it
    
#     if cur.execute(f"SELECT * FROM ddpm WHERE experiment_name='{data[0][0]}' AND checkpoint_name='{data[0][1]}' AND smoothing='{data[0][2]}' AND dataset_name='{data[0][3]}' AND split='{data[0][4]}'").fetchone():
#         print('Deleting existing entry')
#         cur.execute(f"DELETE FROM ddpm WHERE experiment_name='{data[0][0]}' AND checkpoint_name='{data[0][1]}' AND smoothing='{data[0][2]}' AND dataset_name='{data[0][3]}' AND split='{data[0][4]}'")
        
    
#     cur.executemany("INSERT INTO ddpm VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data)
#     con.commit()
    
#     con.close() 

def log_to_database(data, log_subdir):
    
    base_folder = f'/home/s94zalek_hpc/shape_matching/my_code/experiments/ddpm_results/{log_subdir}'
    os.makedirs(base_folder, exist_ok=True)
    
    curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_name = f"{data['experiment_name']}_{data['checkpoint_name']}_{data['smoothing_type']}_{data['smoothing_iter']}_{data['dataset_name']}_{data['split']}_{curr_time}.yaml"

    # save to yaml
    with open(f'{base_folder}/{log_name}', 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    


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
    model = DiagConditionedUnet(config["model_params"])
    
    if "accelerate" in config and config["accelerate"]:
        accelerate.load_checkpoint_in_model(model, f"{exp_base_folder}/checkpoints/{checkpoint_name}/model.safetensors")
    else:
        model.load_state_dict(torch.load(f"{exp_base_folder}/checkpoints/{checkpoint_name}"))
    
    model = model.to('cuda')
    
    ### Sign correction network
    sign_corr_net = diffusion_network.DiffusionNet(
        **config["sign_net"]["net_params"]
        ).to('cuda')
        # in_channels=128,
        # out_channels=config["model_params"]["sample_size"] // config["evecs_per_support"],
        # cache_dir=None,
        # input_type=config["net_input_type"],
        # k_eig=128,
        # n_block=2 
        

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
    # Logging
    ##########################################

    log_dir = f'{exp_base_folder}/eval/{checkpoint_name}/{dataset_name}-{split}/no_smoothing'
    os.makedirs(log_dir, exist_ok=True)

    fig_dir = f'{log_dir}/figs'
    os.makedirs(fig_dir, exist_ok=True)

    log_file_name = f'{log_dir}/log.txt'

    ##########################################

    ratios = []
    geo_errs = []
    geo_errs_zo = []
    
    geo_errs_median = []
    geo_errs_zo_median = []
    geo_errs_median_p2p = []
    geo_errs_dirichlet = []

    Cxy_est_list = []
    C_gt_xy_corr_list = []


    data_range = tqdm(range(len(test_dataset)))

    # data_range = tqdm(range(2))
    # print('!!!!!! Data range limited to 10 !!!!!!!')

    for i in data_range:

        data = test_dataset[i]
        
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'
        
        verts_first = data['first']['verts'].unsqueeze(0).to(device)
        verts_second = data['second']['verts'].unsqueeze(0).to(device)
        
        faces_first = data['first']['faces'].unsqueeze(0).to(device)
        faces_second = data['second']['faces'].unsqueeze(0).to(device)

        evecs_first = data['first']['evecs'][:, :num_evecs].unsqueeze(0).to(device)
        evecs_second = data['second']['evecs'][:, :num_evecs].unsqueeze(0).to(device)
        
        evals_first = data['first']['evals'][:num_evecs]
        evals_second = data['second']['evals'][:num_evecs]

        corr_first = data['first']['corr']
        corr_second = data['second']['corr']
        
        if config["sign_net"]["with_mass"]:
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
                sign_corr_net, verts_first, faces_first, evecs_first, 
                mass_mat=mass_mat_first, input_type=sign_corr_net.input_type,
                # mass=None, L=None, evals=None, evecs=None, gradX=None, gradY=None
                mass=data['first']['mass'].unsqueeze(0), L=data['first']['L'].unsqueeze(0),
                evals=data['first']['evals'][:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                evecs=data['first']['evecs'][:,:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                gradX=data['first']['gradX'].unsqueeze(0), gradY=data['first']['gradY'].unsqueeze(0)
                )
            sign_pred_second, support_vector_norm_second, _ = predict_sign_change(
                sign_corr_net, verts_second, faces_second, evecs_second, 
                mass_mat=mass_mat_second, input_type=sign_corr_net.input_type,
                # mass=None, L=None, evals=None, evecs=None, gradX=None, gradY=None
                mass=data['second']['mass'].unsqueeze(0), L=data['second']['L'].unsqueeze(0),
                evals=data['second']['evals'][:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                evecs=data['second']['evecs'][:,:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                gradX=data['second']['gradX'].unsqueeze(0), gradY=data['second']['gradY'].unsqueeze(0)
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
                data['first']['mass'].unsqueeze(0)
                ).to(device)
            mass_mat_second = torch.diag_embed(
                data['second']['mass'].unsqueeze(0)
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
        


        # gt corrected fmap
        C_gt_xy_corr = torch.linalg.lstsq(
            evecs_second_corrected[corr_second],
            evecs_first_corrected[corr_first]
            ).solution
        
        # gt original fmap
        C_gt_xy = torch.linalg.lstsq(
            evecs_second.cpu()[0, corr_second],
            evecs_first.cpu()[0, corr_first]
            ).solution
        
        
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
            evecs_cond_full = torch.cat(
                (evecs_cond_first.unsqueeze(0), evecs_cond_second.unsqueeze(0)),
                0)
            conditioning = torch.cat((conditioning, evecs_cond_full), 0)
        
        
        ###############################################
        # Sample the model
        ###############################################
        
        # x_sampled = torch.rand(1, 1, model.model.sample_size, model.model.sample_size).to(device)
        # y = conditioning.unsqueeze(0).to(device) 
        
        # x_sampled = torch.rand(args.num_iters_avg, 1, model.model.sample_size, model.model.sample_size).to(device)
        x_sampled = torch.rand(args.num_iters_avg, 1, model.sample_size, model.sample_size).to(device)
        
        # repeat conditioning for each sample, [num_iters_avg, n_channels, model.sample_size, model.sample_size]
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

        # Cxy_est = x_sampled[0][0].cpu()
        
        dist_x = torch.tensor(
            compute_geodesic_distmat(data['first']['verts'].numpy(), data['first']['faces'].numpy())    
        )
        
        geo_err_est_sampled = []
        geo_err_est_zo_sampled = []
        p2p_est_zo_sampled = []
        
        for k in range(args.num_iters_avg):
            
            Cxy_est_k = x_sampled[k][0].cpu()
            
            ###############################################
            # Zoomout
            ###############################################
            
            evecs_first_zo = torch.cat(
                [evecs_first_corrected,
                data['first']['evecs'][:, num_evecs:]], 1)
            evecs_second_zo = torch.cat(
                [evecs_second_corrected,
                data['second']['evecs'][:, num_evecs:]], 1)
            
            C_xy_est_zo_k = zoomout_custom.zoomout(
                FM_12=Cxy_est_k.to(device), 
                evects1=evecs_first_zo.to(device), 
                evects2=evecs_second_zo.to(device),
                nit=evecs_first_zo.shape[1] - num_evecs, step=1,
                # nit=8, step=(evecs_first_zo.shape[1] - num_evecs) // 8,
            ).cpu()
            
            
            
            ###############################################
            # Evaluation
            ###############################################  
            
            # hard correspondence 
            p2p_gt = fmap_util.fmap2pointmap(
                C12=C_gt_xy,
                evecs_x=evecs_first.cpu()[0],
                evecs_y=evecs_second.cpu()[0],
                )
            p2p_corr_gt = fmap_util.fmap2pointmap(
                C12=C_gt_xy_corr,
                evecs_x=evecs_first_corrected,
                evecs_y=evecs_second_corrected,
                )
            p2p_est_k = fmap_util.fmap2pointmap(
                Cxy_est_k,
                evecs_x=evecs_first_corrected,
                evecs_y=evecs_second_corrected,
                )
            p2p_est_zo_k = fmap_util.fmap2pointmap(
                C_xy_est_zo_k,
                evecs_x=evecs_first_zo,
                evecs_y=evecs_second_zo,
                )
            
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
            geo_err_est_zo_k = geodist_metric.calculate_geodesic_error(
                dist_x, data['first']['corr'], data['second']['corr'], p2p_est_zo_k, return_mean=False
                )
            
            geo_err_est_sampled.append(geo_err_est_k.mean())
            geo_err_est_zo_sampled.append(geo_err_est_zo_k.mean())
            
            p2p_est_zo_sampled.append(p2p_est_zo_k)
            
        geo_err_est_sampled = torch.tensor(geo_err_est_sampled)
        geo_err_est_zo_sampled = torch.tensor(geo_err_est_zo_sampled)
        p2p_est_zo_sampled = torch.stack(p2p_est_zo_sampled)
        
        ##########################################################
        # p2p map selection
        ##########################################################
        
        p2p_dirichlet, p2p_median, dirichlet_energy_list = select_p2p_map_dirichlet(
            p2p_est_zo_sampled, verts_first[0].cpu(), data['second']['L'], dist_x,
            num_samples_median=args.num_samples_median
            )
        
        
        geo_err_est_zo_median = geodist_metric.calculate_geodesic_error(
            dist_x, data['first']['corr'], data['second']['corr'], p2p_median, return_mean=True
                )
        geo_err_est_zo_dirichlet = geodist_metric.calculate_geodesic_error(
            dist_x, data['first']['corr'], data['second']['corr'], p2p_dirichlet, return_mean=True
                )
        
        sorted_idxs_geo_err_zo = torch.argsort(geo_err_est_zo_sampled)
            
        # replace code above with writing to log file
        with open(log_file_name, 'a') as f:
            f.write(f'{i}\n')
            f.write(f'Geo error GT: {geo_err_gt.mean() * 100:.2f}\n')
            f.write(f'Geo error GT corr: {geo_err_corr_gt.mean() * 100:.2f}\n')
            # f.write(f'Geo error est: {geo_err_est_sampled.mean() * 100:.2f}\n')
            # f.write(f'Geo error est zo: {geo_err_est_zo_sampled.mean() * 100:.2f}\n')
            # # f.write(f'MSE fmap: {mse_fmap:.3f}\n')
            # # f.write(f'MSE abs fmap: {mse_abs_fmap:.3f}\n')
            # f.write('-----------------------------------\n')
            
            
            f.write(f'Geo error est mean: {geo_err_est_sampled.mean() * 100:.2f}, \n'+\
            f'Geo error est median: {geo_err_est_sampled.median() * 100:.2f}, \n'+\
            f'Geo error est: {geo_err_est_sampled[sorted_idxs_geo_err_zo] * 100}, \n'
            f'Geo error est zo mean: {geo_err_est_zo_sampled.mean() * 100:.2f}, \n'+\
            f'Geo error est zo median: {geo_err_est_zo_sampled.median() * 100:.2f}\n'
            f'Geo error est zo: {geo_err_est_zo_sampled[sorted_idxs_geo_err_zo] * 100}\n'
            f'Geo error est p2p zo median: {geo_err_est_zo_median * 100:.2f}\n'
            f'Geo error est p2p zo dirichlet: {geo_err_est_zo_dirichlet * 100:.2f}\n'
            f'Dirichlet energy: {dirichlet_energy_list[sorted_idxs_geo_err_zo]}\n'
            )
            f.write('-----------------------------------\n')
            
            
        
        # break
        # plt.savefig(f'{fig_dir}/{i}.png')
        # plt.close()
        
        # print(f'{i:2d}) ratio {geo_err_est.mean() / geo_err_corr_gt.mean():.2f}')
        
        ratio_curr = geo_err_est_sampled.mean() / geo_err_corr_gt.mean()
        
        ratios.append(ratio_curr)
        geo_errs.append(geo_err_est_sampled.mean() * 100)
        geo_errs_zo.append(geo_err_est_zo_sampled.mean() * 100)
        
        geo_errs_median.append(geo_err_est_sampled.median() * 100)
        geo_errs_zo_median.append(geo_err_est_zo_sampled.median() * 100)
        
        geo_errs_median_p2p.append(geo_err_est_zo_median * 100)
        geo_errs_dirichlet.append(geo_err_est_zo_dirichlet * 100)
        
        # Cxy_est_list.append(Cxy_est)
        # C_gt_xy_corr_list.append(C_gt_xy_corr)
        
        # data_range.set_description(
        #     f'Geo error est: {geo_err_curr:.2f}, '+\
        #     f'Mean {torch.tensor(geo_errs).mean():.2f}, '+\
        #     f'Median {torch.tensor(geo_errs).median():.2f}, '+\
        #     f'Ratio: {ratio_curr:.2f}, '+\
        #     f'Mean: {torch.tensor(ratios).mean():.2f}, '+\
        #     f'Median: {torch.tensor(ratios).median():.2f}'
        #     )
    

    ratios = torch.tensor(ratios)
    geo_errs = torch.tensor(geo_errs)
    geo_errs_zo = torch.tensor(geo_errs_zo)
    
    geo_errs_median = torch.tensor(geo_errs_median)
    geo_errs_zo_median = torch.tensor(geo_errs_zo_median)
    
    geo_errs_median_p2p = torch.tensor(geo_errs_median_p2p)
    geo_errs_dirichlet = torch.tensor(geo_errs_dirichlet)
        
    # replace code above with writing to log file
    with open(log_file_name, 'a') as f:
        f.write('-----------------------------------\n')
        f.write('Total statistics\n')
        f.write('-----------------------------------\n')
        f.write(f'Zoomout geo err mean: {geo_errs_zo.mean():.2f}\n')
        f.write(f'Zoomout geo err median: {geo_errs_zo.median():.2f}\n')
        f.write(f'Zoomout geo err min: {geo_errs_zo.min():.2f}\n')
        f.write(f'Zoomout geo err max: {geo_errs_zo.max():.2f}\n')        
        f.write('-----------------------------------\n')
        f.write(f'Mean geo err: {geo_errs.mean():.2f}\n')
        f.write(f'Median geo err: {geo_errs.median():.2f}\n')
        f.write(f'Min geo err: {geo_errs.min():.2f}\n')
        f.write(f'Max geo err: {geo_errs.max():.2f}\n')
        f.write('\n')
        f.write(f'Mean ratio: {ratios.mean():.2f}\n')
        f.write(f'Median ratio: {ratios.median():.2f}\n')
        f.write(f'Min ratio: {ratios.min():.2f}\n')
        f.write(f'Max ratio: {ratios.max():.2f}\n')
        f.write('\n')
        f.write(f'Mean geo err p2p median: {geo_errs_median_p2p.mean():.2f}\n')
        f.write(f'Mean geo err dirichlet: {geo_errs_dirichlet.mean():.2f}\n')
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
        geo_errs_median_p2p.mean().item(),
        # zoomout
        geo_errs_zo.mean().item(), geo_errs_zo_median.mean().item(),
        # pred
        geo_errs.mean().item(), geo_errs_median.mean().item()
        ),]
    
    log_to_database(data)
    

    
if __name__ == '__main__':
    run()