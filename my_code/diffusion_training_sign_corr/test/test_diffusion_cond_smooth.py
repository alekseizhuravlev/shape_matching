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

tqdm._instances.clear()


def parse_args():
    parser = argparse.ArgumentParser(description='Train the model')
    
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--checkpoint_name', type=str)
    
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    
    parser.add_argument('--smoothing_type', choices=['laplacian', 'taubin'])
    parser.add_argument('--smoothing_iter', type=int)
    
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
    geo_errs_zo = []
    geo_errs_pairzo = []
    geo_errs_zo_32 = []
    geo_errs_zo_64 = []

    Cxy_est_list = []
    C_gt_xy_corr_list = []

        
    data_range_pair = tqdm(range(len(test_dataset)), desc='Calculating pair fmaps')

    # data_range_pair = tqdm(range(5))

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
        
        x_sampled = torch.rand(1, 1, model.model.sample_size, model.model.sample_size).to(device)
        y = torch.cat(
            (conditioning_first, conditioning_second),
                      0).unsqueeze(0).to(device)    
         
        # Sampling loop
        for t in noise_scheduler.timesteps:

            # Get model pred
            with torch.no_grad():
                residual = model(x_sampled, t,
                                    conditioning=y
                                    ).sample

            # Update sample with step
            x_sampled = noise_scheduler.step(residual, t, x_sampled).prev_sample

        Cxy_est = x_sampled[0][0]
        
        ###############################################
        
        C_gt_xy = torch.linalg.lstsq(
            evecs_second[corr_second],
            evecs_first[corr_first]
            ).solution
        
        C_gt_xy_corr = torch.linalg.lstsq(
            evecs_second_zo[corr_second],
            evecs_first_zo[corr_first]
            ).solution
        
        Cxy_est_pairzo = zoomout_custom.zoomout(
            FM_12=Cxy_est, 
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
        p2p_est = fmap_util.fmap2pointmap(
            Cxy_est,
            evecs_x=evecs_first_zo[:, :num_evecs],
            evecs_y=evecs_second_zo[:, :num_evecs],
            ).cpu()
        p2p_est_pairzo = fmap_util.fmap2pointmap(
            Cxy_est_pairzo,
            evecs_x=evecs_first_zo,
            evecs_y=evecs_second_zo,
            ).cpu()
        
        # distance matrices
        dist_x = torch.cdist(data['first']['verts'], data['first']['verts'])
        dist_y = torch.cdist(data['second']['verts'], data['second']['verts'])

        # geodesic error
        geo_err_gt = geodist_metric.calculate_geodesic_error(
            dist_x, data['first']['corr'], data['second']['corr'], p2p_gt, return_mean=False
            )  
        geo_err_corr_gt = geodist_metric.calculate_geodesic_error(
            dist_x, data['first']['corr'], data['second']['corr'], p2p_corr_gt, return_mean=False
            )
        geo_err_est = geodist_metric.calculate_geodesic_error(
            dist_x, data['first']['corr'], data['second']['corr'], p2p_est, return_mean=False
            )
        geo_err_est_pairzo = geodist_metric.calculate_geodesic_error(
            dist_x, data['first']['corr'], data['second']['corr'], p2p_est_pairzo, return_mean=False
            )
        
        mse_fmap = torch.sum((C_gt_xy_corr[:num_evecs,:num_evecs] - Cxy_est) ** 2).cpu()
        mse_abs_fmap = torch.sum((C_gt_xy_corr[:num_evecs,:num_evecs].abs() - Cxy_est.abs()) ** 2).cpu()
        
        
        # put the data to cpu
        Cxy_est = Cxy_est.cpu()
        C_gt_xy_corr = C_gt_xy_corr.cpu()
        C_gt_xy = C_gt_xy.cpu()
        # Cxy_est_zo = Cxy_est_zo.cpu()
        Cxy_est_pairzo = Cxy_est_pairzo.cpu()

        # fig, axs = plt.subplots(1, 6, figsize=(20, 3))
        
        # l = 0
        # h = num_evecs

        # plotting_utils.plot_Cxy(fig, axs[0], Cxy_est,
        #                         f'Pred, {geo_err_est.mean() * 100:.2f}', l, h, show_grid=False, show_colorbar=False)
        # plotting_utils.plot_Cxy(fig, axs[1], C_gt_xy_corr,
        #                         f'GT corrected, {geo_err_corr_gt.mean() * 100:.2f}', l, h, show_grid=False, show_colorbar=False)
        # plotting_utils.plot_Cxy(fig, axs[2], C_gt_xy,
        #                         f'GT orig, {geo_err_gt.mean() * 100:.2f}', l, h, show_grid=False, show_colorbar=False)
        # plotting_utils.plot_Cxy(fig, axs[3], Cxy_est - C_gt_xy_corr[:num_evecs,:num_evecs],
        #                         f'Error', l, h, show_grid=False, show_colorbar=False)
        # plotting_utils.plot_Cxy(fig, axs[4], Cxy_est_zo[:num_evecs, :num_evecs],
        #                         f'After ZO, {geo_err_est_zo.mean() * 100:.2f}', l, h, show_grid=False, show_colorbar=False)
        # plotting_utils.plot_Cxy(fig, axs[5], Cxy_est_zo[:num_evecs, :num_evecs] - C_gt_xy_corr[:num_evecs,:num_evecs],
        #                         f'Error ZO', l, h, show_grid=False, show_colorbar=False)
        
        # plotting_utils.plot_Cxy(fig, axs[4], Cxy_est.abs() - C_gt_xy_corr.abs(),
        #                         f'Error abs', l, h, show_grid=False, show_colorbar=False)
        # plotting_utils.plot_Cxy(fig, axs[6], Cxy_est_zo_first,
        #                         f'Cxy_est_zo_first', l, h, show_grid=False, show_colorbar=False)
        # plotting_utils.plot_Cxy(fig, axs[7], Cxy_est_zo_second,
        #                         f'Cxy_est_zo_second', l, h, show_grid=False, show_colorbar=False)
        
        # replace code above with writing to log file
        with open(log_file_name, 'a') as f:
            f.write(f'{i}\n')
            f.write(f'Geo error GT: {geo_err_gt.mean() * 100:.2f}\n')
            f.write(f'Geo error GT corr: {geo_err_corr_gt.mean() * 100:.2f}\n')
            f.write(f'Geo error est: {geo_err_est.mean() * 100:.2f}\n')
            f.write(f'Geo error est pairzo: {geo_err_est_pairzo.mean() * 100:.2f}\n')
            f.write(f'MSE fmap: {mse_fmap:.3f}\n')
            f.write(f'MSE abs fmap: {mse_abs_fmap:.3f}\n')
            f.write('-----------------------------------\n')
        
        # break
        # plt.savefig(f'{fig_dir}/{i}.png')
        # plt.close()
        
        # print the stats instead of writing to file
        # print(f'{i}')
        # print(f'Geo error GT: {geo_err_gt.mean() * 100:.2f}')
        # print(f'Geo error GT corr: {geo_err_corr_gt.mean() * 100:.2f}')
        # print(f'Geo error est: {geo_err_est.mean() * 100:.2f}')
        # print(f'Geo error est zo: {geo_err_est_zo.mean() * 100:.2f}')
        # print(f'MSE fmap: {mse_fmap:.3f}')
        # print(f'MSE abs fmap: {mse_abs_fmap:.3f}')
        # print('-----------------------------------')
        
        # plt.show()
        # plt.close()
        
        # print(f'{i:2d}) ratio {geo_err_est.mean() / geo_err_corr_gt.mean():.2f}')
        
        ratio_curr = geo_err_est.mean() / geo_err_corr_gt.mean()
        
        ratios.append(ratio_curr)
        geo_errs.append(geo_err_est.mean() * 100)
        geo_errs_pairzo.append(geo_err_est_pairzo.mean() * 100)
        # Cxy_est_list.append(Cxy_est)
        # C_gt_xy_corr_list.append(C_gt_xy_corr)
        
        # if i % 10 == 0:
        data_range_pair.set_description(
            f'Mean {torch.tensor(geo_errs).mean():.2f}, '+\
            f'pairzo {torch.tensor(geo_errs_pairzo).mean():.2f}, '
            )


    ratios = torch.tensor(ratios)
    geo_errs = torch.tensor(geo_errs)
    geo_errs_pairzo = torch.tensor(geo_errs_pairzo)
        
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
        f.write('\n')
        f.write(f'Mean ratio: {ratios.mean():.2f}\n')
        f.write(f'Median ratio: {ratios.median():.2f}\n')
        f.write(f'Min ratio: {ratios.min():.2f}\n')
        f.write(f'Max ratio: {ratios.max():.2f}\n')
        f.write('\n')
        f.write('-----------------------------------\n')