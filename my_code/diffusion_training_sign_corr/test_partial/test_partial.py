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
import accelerate
import my_code.sign_canonicalization.test_sign_correction as test_sign_correction
import networks.fmap_network as fmap_network
from my_code.utils.median_p2p_map import dirichlet_energy

tqdm._instances.clear()

class RegularizedFMNet(torch.nn.Module):
    """Compute the functional map matrix representation in DPFM"""
    def __init__(self, lmbda=0.01, resolvant_gamma=0.5, bidirectional=False):
        super(RegularizedFMNet, self).__init__()
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma
        self.bidirectional = bidirectional

    def compute_functional_map(self, A, B, evals_x, evals_y):
        # A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        # B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = fmap_network.get_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]

        A_t = A.transpose(1, 2)  # [B, C, K]
        A_A_t = torch.bmm(A, A_t)  # [B, K, K]
        B_A_t = torch.bmm(B, A_t)  # [B, K, K]

        C_i = []
        for i in range(evals_x.shape[1]):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.shape[0])], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + self.lmbda * D_i), B_A_t[:, [i], :].transpose(1, 2))
            C_i.append(C.transpose(1, 2))

        Cxy = torch.cat(C_i, dim=1)
         
        return Cxy
    



def get_geo_error(
    p2p_first, p2p_second,
    evecs_first, evecs_second,
    corr_first, corr_second,
    num_evecs, apply_zoomout,
    dist_x,
    regularized=False,
    evecs_trans_first=None, evecs_trans_second=None,
    evals_first=None, evals_second=None,
    return_p2p=False, return_Cxy=False,
    A2=None, fmnet=None
    ):
        
    if regularized:
        Cxy = fmnet.compute_functional_map(
            evecs_trans_second[:num_evecs, p2p_second].unsqueeze(0),
            evecs_trans_first[:num_evecs, p2p_first].unsqueeze(0),
            evals_second[:num_evecs].unsqueeze(0),
            evals_first[:num_evecs].unsqueeze(0), 
        )[0].T
        
    else:
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
            A2=A2
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
    
    # if return_p2p:
    #     return geo_err * 100, p2p
    # else:
    #     return geo_err * 100
    
    if not return_p2p and not return_Cxy:
        return geo_err * 100
    
    payload = [geo_err * 100]
    
    if return_p2p:
        payload.append(p2p)
    if return_Cxy:
        payload.append(Cxy)
        
    return payload


def filter_p2p_by_confidence(
        p2p_first, p2p_second,
        confidence_scores_first, confidence_scores_second,
        confidence_threshold, log_file_name
    ):
    
    assert p2p_first.shape[0] == p2p_second.shape[0]
    
    # select points with both confidence scores above threshold
    valid_points = (confidence_scores_first < confidence_threshold) & (confidence_scores_second < confidence_threshold)
    
    with open(log_file_name, 'a') as f:
        
        while valid_points.sum() < 0.05 * len(valid_points):
            confidence_threshold += 0.05
            valid_points = (confidence_scores_first < confidence_threshold) & (confidence_scores_second < confidence_threshold)
            
            f.write(f'Increasing confidence threshold: {confidence_threshold}\n')
        f.write(f'Valid points: {valid_points.sum()}\n')
        assert valid_points.sum() > 0, "No valid points found"
        
    p2p_first = p2p_first[valid_points]
    p2p_second = p2p_second[valid_points]
    
    return p2p_first, p2p_second


def get_fmaps_evec_signs(
        data, model,
        noise_scheduler, config, args,
        template_shape, sign_corr_net
    ):
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_evecs = config["model_params"]["sample_size"]

        
    verts_first = template_shape['verts'].unsqueeze(0).to(device)
    verts_second = data['verts'].unsqueeze(0).to(device)
    
    faces_first = template_shape['faces'].unsqueeze(0).to(device)
    faces_second = data['faces'].unsqueeze(0).to(device)

    evecs_first = template_shape['evecs'][:, :num_evecs].unsqueeze(0).to(device)
    evecs_second = data['evecs'][:, :num_evecs].unsqueeze(0).to(device)
    

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
        
    # shot descriptors
    if sign_corr_net.input_type == 'shot':
        
        import pyshot       
        
        if 'shot' in template_shape:
            input_feats_first = template_shape['shot'].unsqueeze(0).to(device)
        else:
            input_feats_first = torch.tensor(
                pyshot.get_descriptors(
                template_shape['verts'].numpy().astype(np.double),
                template_shape['faces'].numpy().astype(np.int64),
                radius=100,
                local_rf_radius=100,
                
                min_neighbors=3,
                n_bins=10,
                double_volumes_sectors=True,
                use_interpolation=True,
                use_normalization=True,
            ), dtype=torch.float32).unsqueeze(0).to(device)
        

        if 'shot' in data:
            input_feats_second = data['shot'].unsqueeze(0).to(device)
        else:
            input_feats_second = torch.tensor(
                pyshot.get_descriptors(
                data['verts'].numpy().astype(np.double),
                data['faces'].numpy().astype(np.int64),
                radius=100,
                local_rf_radius=100,
                
                min_neighbors=3,
                n_bins=10,
                double_volumes_sectors=True,
                use_interpolation=True,
                use_normalization=True,
            ), dtype=torch.float32).unsqueeze(0).to(device)
    else:
        input_feats_first = None
        input_feats_second = None
        
        


    ###############################################
    # get conditioning and signs num_iters_avg times
    ###############################################

    evecs_cond_first_list = []
    evecs_cond_second_list = []
    evecs_first_signs_list = []
    evecs_second_signs_list = []

    for _ in range(args.num_iters_avg):

        # predict the sign change
        with torch.no_grad():
            sign_pred_first, support_vector_norm_first, _ = predict_sign_change(
                sign_corr_net, verts_first, faces_first, evecs_first, 
                mass_mat=mass_mat_first, input_type=sign_corr_net.input_type,
                evecs_per_support=config["sign_net"]["evecs_per_support"],
                
                input_feats=input_feats_first,
                
                mass=template_shape['mass'].unsqueeze(0), L=template_shape['L'].unsqueeze(0),
                evals=template_shape['evals'][:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                evecs=template_shape['evecs'][:,:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                gradX=template_shape['gradX'].unsqueeze(0), gradY=template_shape['gradY'].unsqueeze(0)
                )
            sign_pred_second, support_vector_norm_second, _ = predict_sign_change(
                sign_corr_net, verts_second, faces_second, evecs_second, 
                mass_mat=mass_mat_second, input_type=sign_corr_net.input_type,
                evecs_per_support=config["sign_net"]["evecs_per_support"],
                
                input_feats=input_feats_second,
                
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
            
        evecs_cond_first_list.append(evecs_cond_first)
        evecs_cond_second_list.append(evecs_cond_second)
        evecs_first_signs_list.append(torch.sign(sign_pred_first).cpu())
        evecs_second_signs_list.append(torch.sign(sign_pred_second).cpu())
        
    evecs_cond_first_list = torch.stack(evecs_cond_first_list)
    evecs_cond_second_list = torch.stack(evecs_cond_second_list)
    evecs_first_signs_list = torch.stack(evecs_first_signs_list)
    evecs_second_signs_list = torch.stack(evecs_second_signs_list)    
    
    
    ###############################################
    # Conditioning
    ###############################################

    conditioning = torch.cat(
        (evecs_cond_first_list.unsqueeze(1), evecs_cond_second_list.unsqueeze(1)),
        1)
    
    ###############################################
    # Sample the model
    ###############################################
    
    x_sampled = torch.rand(args.num_iters_avg, 1, 
                        config["model_params"]["sample_size"],
                        config["model_params"]["sample_size"]).to(device)
    y = conditioning.to(device)    
        
    # Sampling loop
    for t in noise_scheduler.timesteps:

        # Get model pred
        with torch.no_grad():
            residual = model(x_sampled, t,
                                conditioning=y
                                ).sample

        # Update sample with step
        x_sampled = noise_scheduler.step(residual, t, x_sampled).prev_sample
        
    return x_sampled, evecs_first_signs_list, evecs_second_signs_list


def get_p2p_maps_template(
        data,
        C_yx_est_i, evecs_first_signs_i, evecs_second_signs_i,
        template_shape, args, log_file_name, config
    ):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_evecs = config["model_params"]["sample_size"]
    
    f = open(log_file_name, 'a', buffering=1)
    
    verts_second = data['verts']
    faces_second = data['faces']
    
    evecs_first = template_shape['evecs'][:, :num_evecs]
    evecs_second = data['evecs'][:, :num_evecs]
    
    # evecs_first = template_shape['evecs']
    # evecs_second = data['evecs']
    
    dist_second = torch.tensor(
        compute_geodesic_distmat(
            verts_second.numpy(),
            faces_second.numpy())    
    )
    
    ##########################################################
    # Convert fmaps to p2p maps to template
    ##########################################################
    
    p2p_est = []
    
    # version without zoomout and dirichlet energy condition
    # for k in range(args.num_iters_avg):

    #     evecs_first_corrected = evecs_first * evecs_first_signs_i[k]
    #     evecs_second_corrected = evecs_second * evecs_second_signs_i[k]
    #     Cyx_est_k = C_yx_est_i[k][0].cpu()

    #     p2p_est_k = fmap_util.fmap2pointmap(
    #         C12=Cyx_est_k.to(device),
    #         evecs_x=evecs_second_corrected.to(device),
    #         evecs_y=evecs_first_corrected.to(device),
    #         ).cpu()

    #     p2p_est.append(p2p_est_k)
                
                
    for k in range(args.num_iters_avg):
        
        evecs_first_corrected = evecs_first * evecs_first_signs_i[k]
        evecs_second_corrected = evecs_second * evecs_second_signs_i[k]
        Cyx_est_k = C_yx_est_i[k][0].cpu()
    
        fmap_dimension_k = num_evecs
    
        zo_num_evecs = args.zoomout_num_evecs_template
        if zo_num_evecs is not None and zo_num_evecs > 0 and fmap_dimension_k < zo_num_evecs:
            
            evecs_first_zo = torch.cat(
                [evecs_first_corrected, evecs_first[:, fmap_dimension_k:zo_num_evecs]],
                dim=1
            ).to(device)
            
            evecs_second_zo = torch.cat(
                [evecs_second_corrected, evecs_second[:, fmap_dimension_k:zo_num_evecs]],
                dim=1                    
            ).to(device)
            
            
            Cyx_zo_k = zoomout_custom.zoomout(
                FM_12=Cyx_est_k.to(device), 
                evects1=evecs_second_zo,
                evects2=evecs_first_zo,
                nit=zo_num_evecs-fmap_dimension_k, step=1,
                A2=template_shape['mass'].to(device),
            )
            p2p_zo_k = fmap_util.fmap2pointmap(
                C12=Cyx_zo_k,
                evecs_x=evecs_second_zo,
                evecs_y=evecs_first_zo,
                ).cpu()
            
            # dirichlet_energy_zo = dirichlet_energy(p2p_zo_k, verts_second, template_shape['L'])
            # f.write(f'Zoomout energy: {dirichlet_energy_zo}\n')
            
            p2p_est_k = p2p_zo_k
            
        else:
            p2p_est_k = fmap_util.fmap2pointmap(
                C12=Cyx_est_k.to(device),
                evecs_x=evecs_second_corrected.to(device),
                evecs_y=evecs_first_corrected.to(device),
                ).cpu()
            
        p2p_est.append(p2p_est_k)
                
                
                
                

    p2p_est = torch.stack(p2p_est)
        
    ##########################################################
    # p2p map selection
    ##########################################################
    
    p2p_dirichlet, p2p_median, confidence_scores, dirichlet_energy_list = select_p2p_map_dirichlet(
        p2p_est,
        verts_second,
        template_shape['L'], 
        dist_second,
        num_samples_median=args.num_samples_median
        )
         
    f.write(f'Template stage\n')
    f.write(f'Dirichlet energy: {dirichlet_energy_list}\n')
    f.write(f'Confidence scores: {confidence_scores}\n')
    f.write(f'Mean confidence score: {confidence_scores.mean():.3f}\n')
    f.write(f'Median confidence score: {confidence_scores.median():.3f}\n')
    f.write('\n')
    
    # replace the code above with print, remove \n at the end
    # print(f'Template stage')
    # print(f'Dirichlet energy: {dirichlet_energy_list}')
    # print(f'Confidence scores: {confidence_scores}')
    # print(f'Mean confidence score: {confidence_scores.mean():.3f}')
    # print(f'Median confidence score: {confidence_scores.median():.3f}')
        
    f.close()
        
    return p2p_est, p2p_dirichlet, p2p_median, confidence_scores, dist_second


def zoomout_after_reindexing(
    p2p_input,
    evecs_first, evecs_second,
    A2, start_dim,
):

    if A2 is not None:
        if A2.ndim == 1:
            Cxy = evecs_second[:, :start_dim].T @ (A2[:, None] * evecs_first[p2p_input, :start_dim])
        else:
            Cxy = evecs_second[:, :start_dim].T @ (A2 @ evecs_first[p2p_input, :start_dim])
    else:
        Cxy = fmap_util.pointmap2fmap(p2p_input, evecs_second[:, :start_dim], evecs_first[:, :start_dim])

    Cxy = zoomout_custom.zoomout(
        FM_12=Cxy, 
        evects1=evecs_first,
        evects2=evecs_second,
        nit=evecs_first.shape[1] - start_dim, step=1,
        A2=A2
    )
    
    num_evecs = evecs_first.shape[1]
        
    p2p = fmap_util.fmap2pointmap(
        C12=Cxy,
        evecs_x=evecs_first[:, :num_evecs],
        evecs_y=evecs_second[:, :num_evecs],
        )
    
    return p2p.cpu(), Cxy.cpu()


def run():

    args = parse_args()

    # args = Arguments()

    # configuration
    experiment_name = args.experiment_name
    checkpoint_name = args.checkpoint_name

    ### config
    exp_base_folder = f'/home/s94zalek_hpc/shape_matching/my_code/experiments/ddpm/{experiment_name}'
    with open(f'{exp_base_folder}/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    ### model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DiagConditionedUnet(config["model_params"])

    if "accelerate" in config and config["accelerate"]:
        accelerate.load_checkpoint_in_model(model, f"{exp_base_folder}/checkpoints/{checkpoint_name}/model.safetensors")
    else:
        model.load_state_dict(torch.load(f"{exp_base_folder}/checkpoints/{checkpoint_name}"))

    model.to(device)

    ### Sign correction network
    sign_corr_net = diffusion_network.DiffusionNet(
        **config["sign_net"]["net_params"]
        )        
    sign_corr_net.load_state_dict(torch.load(
            f'{config["sign_net"]["net_path"]}/{config["sign_net"]["n_iter"]}.pth'
            ))
    sign_corr_net.to(device)


    ### noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                                    clip_sample=True) 

    # fmap network
    fmnet = RegularizedFMNet(lmbda=config["sign_net"]["regularization_lambda"], resolvant_gamma=0.5)


    ### test dataset
    dataset_name = args.dataset_name
    split = args.split

    single_dataset, test_dataset = data_loading.get_val_dataset(
        dataset_name, split, 200, preload=False, return_evecs=True, centering='mean'
        )
    # sign_corr_net.cache_dir = single_dataset.lb_cache_dir

    num_evecs = config["model_params"]["sample_size"]

    ##########################################
    # Template
    ##########################################

    template_shape = template_dataset.get_template(
        num_evecs=200,
        centering='mean',
        template_path=f'/home/s94zalek_hpc/shape_matching/data/SURREAL_full/template/{config["sign_net"]["template_type"]}/template.off',
        template_corr=np.loadtxt(
            f'/home/s94zalek_hpc/shape_matching/data/SURREAL_full/template/{config["sign_net"]["template_type"]}/corr.txt',
            dtype=np.int32) - 1,
        return_shot=config["sign_net"]['net_params']['input_type'] == 'shot',
        )    

    ##########################################
    # Logging
    ##########################################

    if args.smoothing_type is not None:
        test_name = f'{args.smoothing_type}-{args.smoothing_iter}'
    else:
        test_name = 'no_smoothing'

    log_dir = f'{exp_base_folder}/eval/{checkpoint_name}/{dataset_name}-{split}/{test_name}'
    os.makedirs(log_dir, exist_ok=True)

    fig_dir = f'{log_dir}/figs'
    os.makedirs(fig_dir, exist_ok=True)

    log_file_name = f'{log_dir}/log_{test_name}.txt'



    ##########################################
    # 1.1: Template stage, get the functional maps and signs of evecs
    ##########################################


    if args.reduced:
        data_range_2 = [6]
        print('!!! WARNING: only 2 samples are processed !!!')
        
    else:
        data_range_2 = range(len(test_dataset))


        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    geo_errs_gt = []
    geo_errs_pairzo = []
    geo_errs_dirichlet = []
    geo_errs_median = []
    geo_errs_median_filtered = []
    geo_errs_median_filtered_noZo = []
    geo_errs_dirichlet_pairzo = []
    geo_errs_median_pairzo = []


    for i in tqdm(data_range_2, desc='Calculating pair fmaps'):
        
        data = test_dataset[i]        
        
        evecs_first = data['first']['evecs'][:, :].to(device)
        evecs_second = data['second']['evecs'][:, :].to(device)
        
        evecs_trans_first = data['first']['evecs_trans'][:, :].to(device)
        evecs_trans_second = data['second']['evecs_trans'][:, :].to(device)
        
        evals_first = data['first']['evals'][:num_evecs].to(device)
        evals_second = data['second']['evals'][:num_evecs].to(device)

        corr_first = data['first']['corr'].to(device)
        corr_second = data['second']['corr'].to(device)
        
        mass_second = data['second']['mass'].to(device)
        
        ###############################################
        # Functional maps
        ###############################################
        
        # first mesh
        
            
        C_sampled_first_list, evecs_first_signs_list_first, evecs_second_signs_list_first = get_fmaps_evec_signs(
            data['first'], model,
            noise_scheduler, config, args,
            template_shape, sign_corr_net
        )
        
        if config["fmap_direction"] == 'xy':
            Cxy_first_list = C_sampled_first_list
            Cyx_first_list = Cxy_first_list.transpose(2, 3)
        else:
            Cyx_first_list = C_sampled_first_list
            Cxy_first_list = Cyx_first_list.transpose(2, 3)
        
        p2p_est_first, p2p_dirichlet_first, p2p_median_first, confidence_scores_first, dist_x = get_p2p_maps_template(
            data['first'],
            Cyx_first_list, evecs_first_signs_list_first, evecs_second_signs_list_first,
            template_shape, args, log_file_name, config
        )
        # p2p_est_first_rev, p2p_dirichlet_first_rev, p2p_median_first_rev, _, _ = get_p2p_maps_template(
        #     template_shape,
        #     Cxy_first_list, evecs_second_signs_list_first, evecs_first_signs_list_first,
        #     data['first'], args, log_file_name, config
        # )
        
        
        # second mesh
        
        C_sampled_second_list, evecs_first_signs_list_second, evecs_second_signs_list_second = get_fmaps_evec_signs(
            data['second'], model,
            noise_scheduler, config, args,
            template_shape, sign_corr_net
        )
        
        if config["fmap_direction"] == 'xy':
            Cxy_second_list = C_sampled_second_list
            Cyx_second_list = Cxy_second_list.transpose(2, 3)
        else:
            Cyx_second_list = C_sampled_second_list
            Cxy_second_list = Cyx_second_list.transpose(2, 3)
        
        # p2p_est_second, p2p_dirichlet_second, p2p_median_second, confidence_scores_second, dist_y = get_p2p_maps_template(
        #     data['second'],
        #     Cyx_second_list, evecs_first_signs_list_second, evecs_second_signs_list_second,
        #     template_shape, args, log_file_name, config
        # )
        
        p2p_est_second_rev, p2p_dirichlet_second_rev, p2p_median_second_rev, confidence_scores_second_rev, _ = get_p2p_maps_template(
            template_shape,
            Cxy_second_list, evecs_second_signs_list_second, evecs_first_signs_list_second,
            data['second'], args, log_file_name, config
        )
        
            
        ###############################################
        # Reflected correspondences
        ###############################################
        
        print('!!! using reflected correspondences')
        
        verts_first_orig = data['first']['verts']

        verts_first_reflected = verts_first_orig.clone()
        verts_first_reflected[:, 0] *= -1

        symm_map_first = fmap_util.nn_query(
            verts_first_reflected,
            verts_first_orig
            )    
        
        corr_first_symm = symm_map_first[corr_first.cpu()]
        
        ###############################################

        corr_first = corr_first.cpu()
        corr_second = corr_second.cpu()
        

        p2p_est_pairzo = []
        geo_err_est_pairzo = []
        Cxy_est_pairzo = []

        for k in range(args.num_iters_avg):
            
            p2p_est_pairzo_k = p2p_est_first[k][p2p_est_second_rev[k]].cpu()
            
            # p2p_est_pairzo_k, Cxy_est_pairzo_k = zoomout_after_reindexing(
            #     p2p_est_pairzo_k.to(device),
            #     evecs_first.to(device),
            #     evecs_second.to(device),
            #     mass_second.to(device), 
            #     num_evecs,
            # )
            
            # p2p_est_pairzo_k, Cxy_est_pairzo_k = zoomout_after_reindexing(
            #     p2p_est_pairzo_k.to(device),
            #     evecs_first.to(device), evecs_second.to(device),
            #     evecs_trans_first.to(device), evecs_trans_second.to(device),
            #     evals_first.to(device), evals_second.to(device), 
            #     args.reduced_dim, fmnet
            # )   
            
            # Cxy_est_pairzo.append(Cxy_est_pairzo_k)
            p2p_est_pairzo.append(p2p_est_pairzo_k)
            
            geo_err_est_pairzo.append(
                min(
                    geodist_metric.calculate_geodesic_error(
                        dist_x, corr_first, corr_second, p2p_est_pairzo_k, return_mean=True
                    ) * 100,
                    geodist_metric.calculate_geodesic_error(
                        dist_x, corr_first_symm, corr_second, p2p_est_pairzo_k, return_mean=True
                    ) * 100,
                ))

        p2p_est_pairzo = torch.stack(p2p_est_pairzo)
        geo_err_est_pairzo = torch.tensor(geo_err_est_pairzo)



        p2p_est_dirichlet = p2p_dirichlet_first[p2p_dirichlet_second_rev].cpu()
        # p2p_est_dirichlet, _ = zoomout_after_reindexing(
        #     p2p_est_dirichlet.to(device),
        #     evecs_first.to(device),
        #     evecs_second.to(device),
        #     mass_second.to(device), 
        #     num_evecs,
        # )
        geo_err_est_dirichlet = min(
            geodist_metric.calculate_geodesic_error(
                dist_x, corr_first, corr_second, p2p_est_dirichlet, return_mean=True
            ) * 100,
            geodist_metric.calculate_geodesic_error(
                dist_x, corr_first_symm, corr_second, p2p_est_dirichlet, return_mean=True
            ) * 100
        )

        p2p_est_median = p2p_median_first[p2p_median_second_rev].cpu()
        # p2p_est_median, _ = zoomout_after_reindexing(
        #     p2p_est_median.to(device),
        #     evecs_first.to(device),
        #     evecs_second.to(device),
        #     mass_second.to(device), 
        #     num_evecs,
        # )
        geo_err_est_median = min(
            geodist_metric.calculate_geodesic_error(
                dist_x, corr_first, corr_second, p2p_est_median, return_mean=True
            ) * 100,
            geodist_metric.calculate_geodesic_error(
                dist_x, corr_first_symm, corr_second, p2p_est_median, return_mean=True
            ) * 100, 
        )
        
        
        
        p2p_dirichlet_pairzo, p2p_median_pairzo, confidence_scores, dirichlet_energy_list = select_p2p_map_dirichlet(
            p2p_est_pairzo,
            data['first']['verts'],
            data['second']['L'], 
            dist_x,
            num_samples_median=args.num_samples_median
            )
        # p2p_dirichlet_pairzo, _ = zoomout_after_reindexing(
        #     p2p_dirichlet_pairzo.to(device),
        #     evecs_first.to(device),
        #     evecs_second.to(device),
        #     mass_second.to(device), 
        #     num_evecs,
        # )
        # p2p_median_pairzo, _ = zoomout_after_reindexing(
        #     p2p_median_pairzo.to(device),
        #     evecs_first.to(device),
        #     evecs_second.to(device),
        #     mass_second.to(device), 
        #     num_evecs,
        # )

        geo_err_dirichlet_pairzo = min(
            geodist_metric.calculate_geodesic_error(
                dist_x, corr_first, corr_second, p2p_dirichlet_pairzo, return_mean=True
            ) * 100,
            geodist_metric.calculate_geodesic_error(
                dist_x, corr_first_symm, corr_second, p2p_dirichlet_pairzo, return_mean=True
            ) * 100
        )
        
        geo_err_median_pairzo = min(
            geodist_metric.calculate_geodesic_error(
                dist_x, corr_first, corr_second, p2p_median_pairzo, return_mean=True
            ) * 100,
            geodist_metric.calculate_geodesic_error(
                dist_x, corr_first_symm, corr_second, p2p_median_pairzo, return_mean=True
            ) * 100,
        )
             
        


        # corr_first = corr_first.cpu()
        # corr_second = corr_second.cpu()
        

        # p2p_est_pairzo = []
        # geo_err_est_pairzo = []

        # for k in range(args.num_iters_avg):
            
        #     p2p_est_pairzo_k = p2p_est_first[k][p2p_est_second_rev[k]].cpu()
            
        #     p2p_est_pairzo.append(p2p_est_pairzo_k)
        #     geo_err_est_pairzo.append(
        #         geodist_metric.calculate_geodesic_error(
        #         dist_x, corr_first, corr_second, p2p_est_pairzo_k, return_mean=True
        #     ) * 100)
    
        # p2p_est_pairzo = torch.stack(p2p_est_pairzo)
        # geo_err_est_pairzo = torch.tensor(geo_err_est_pairzo)



        # p2p_est_dirichlet = p2p_dirichlet_first[p2p_dirichlet_second_rev].cpu()
        # geo_err_est_dirichlet = geodist_metric.calculate_geodesic_error(
        #     dist_x, corr_first, corr_second, p2p_est_dirichlet, return_mean=True
        # ) * 100

        # p2p_est_median = p2p_median_first[p2p_median_second_rev].cpu()
        # geo_err_est_median = geodist_metric.calculate_geodesic_error(
        #     dist_x, corr_first, corr_second, p2p_est_median, return_mean=True
        # ) * 100
        
        
        
        # p2p_dirichlet_pairzo, p2p_median_pairzo, confidence_scores, dirichlet_energy_list = select_p2p_map_dirichlet(
        #     p2p_est_pairzo,
        #     data['first']['verts'],
        #     data['second']['L'], 
        #     dist_x,
        #     num_samples_median=args.num_samples_median
        #     )
        
        # geo_err_dirichlet_pairzo = geodist_metric.calculate_geodesic_error(
        #     dist_x, corr_first, corr_second, p2p_dirichlet_pairzo, return_mean=True
        # ) * 100
        # geo_err_median_pairzo = geodist_metric.calculate_geodesic_error(
        #     dist_x, corr_first, corr_second, p2p_median_pairzo, return_mean=True
        # ) * 100
        
        
        with open(log_file_name, 'a') as f:
            
            if "id" in data["first"] and "id" in data["second"]:
                f.write(f'{i}: {data["first"]["id"]}, {data["second"]["id"]}\n')
            else:
                # print "name" instead of "id"
                f.write(f'{i}: {data["first"]["name"]}, {data["second"]["name"]}\n')

            f.write(f'Geo error est pairzo: {geo_err_est_pairzo}\n')
            f.write(f'Geo error est pairzo mean: {geo_err_est_pairzo.mean():.2f}\n')
            f.write(f'Geo error est dirichlet: {geo_err_est_dirichlet:.2f}\n')
            f.write(f'Geo error est median: {geo_err_est_median:.2f}\n')
            f.write(f'Geo error dirichlet pairzo: {geo_err_dirichlet_pairzo:.2f}\n')
            f.write(f'Geo error median pairzo: {geo_err_median_pairzo:.2f}\n')
            f.write('-----------------------------------\n')
        
        geo_errs_pairzo.append(geo_err_est_pairzo.mean())
        geo_errs_dirichlet.append(geo_err_est_dirichlet)
        geo_errs_median.append(geo_err_est_median)
        geo_errs_dirichlet_pairzo.append(geo_err_dirichlet_pairzo)
        geo_errs_median_pairzo.append(geo_err_median_pairzo)
        
        
    # geo_errs_gt = torch.tensor(geo_errs_gt)
    geo_errs_pairzo = torch.tensor(geo_errs_pairzo)
    geo_errs_dirichlet = torch.tensor(geo_errs_dirichlet)
    geo_errs_median = torch.tensor(geo_errs_median)
    # geo_errs_median_filtered = torch.tensor(geo_errs_median_filtered)
    # geo_errs_median_filtered_noZo = torch.tensor(geo_errs_median_filtered_noZo)
    geo_errs_dirichlet_pairzo = torch.tensor(geo_errs_dirichlet_pairzo)
    geo_errs_median_pairzo = torch.tensor(geo_errs_median_pairzo)
    
    
    data_to_log = {
        # 'experiment_name': args.experiment_name,
        # 'checkpoint_name': args.checkpoint_name, 
        # 'smoothing': f'{args.smoothing_type}-{args.smoothing_iter}' if args.smoothing_type is not None else 'no_smoothing',
        # 'dataset_name': args.dataset_name,
        # 'split': args.split,
        
        **vars(args),
        
        # 'confidence_filtered': geo_errs_median_filtered.mean().item(),
        
        'dirichlet': geo_errs_dirichlet.mean().item(),
        'p2p_median': geo_errs_median.mean().item(),
        
        'zoomout_mean': geo_errs_pairzo.mean().item(),
        'zoomout_median': geo_errs_pairzo.median().item(),
        
        # 'filtered_noZo': geo_errs_median_filtered_noZo.mean().item(),
        'dirichlet_pairzo': geo_errs_dirichlet_pairzo.mean().item(),
        'median_pairzo': geo_errs_median_pairzo.mean().item(),
        }
    
    
    log_to_database(data_to_log, args.log_subdir)
        

if __name__ == '__main__':
    run()
    