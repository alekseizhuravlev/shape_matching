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
from my_code.diffusion_training_sign_corr.test.test_diffusion_cond import select_p2p_map_dirichlet, log_to_database
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
                
                mass=template_shape['mass'].unsqueeze(0), L=template_shape['L'].unsqueeze(0),
                evals=template_shape['evals'][:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                evecs=template_shape['evecs'][:,:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                gradX=template_shape['gradX'].unsqueeze(0), gradY=template_shape['gradY'].unsqueeze(0)
                )
            sign_pred_second, support_vector_norm_second, _ = predict_sign_change(
                sign_corr_net, verts_second, faces_second, evecs_second, 
                mass_mat=mass_mat_second, input_type=sign_corr_net.input_type,
                evecs_per_support=config["sign_net"]["evecs_per_support"],
                
                mass=data['mass'].unsqueeze(0), L=data['L'].unsqueeze(0),
                evals=data['evals'][:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                evecs=data['evecs'][:,:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                gradX=data['gradX'].unsqueeze(0), gradY=data['gradY'].unsqueeze(0)
                )

        # correct the evecs
        evecs_first_corrected = evecs_first.cpu()[0] * torch.sign(sign_pred_first).cpu()
        # evecs_first_corrected_norm = evecs_first_corrected / torch.norm(evecs_first_corrected, dim=0, keepdim=True)
        evecs_first_corrected_norm = torch.nn.functional.normalize(evecs_first_corrected, p=2, dim=0)
        
        evecs_second_corrected = evecs_second.cpu()[0] * torch.sign(sign_pred_second).cpu()
        # evecs_second_corrected_norm = evecs_second_corrected / torch.norm(evecs_second_corrected, dim=0, keepdim=True)
        evecs_second_corrected_norm = torch.nn.functional.normalize(evecs_second_corrected, p=2, dim=0)
        

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
        template_shape, args, log_file_name, config,
        apply_zoomout
    ):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_evecs = config["model_params"]["sample_size"]
    
    f = open(log_file_name, 'a', buffering=1)
    
    verts_second = data['verts']
    faces_second = data['faces']
    
    evecs_first = template_shape['evecs']
    evecs_second = data['evecs']
    
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
    for k in range(args.num_iters_avg):

        evecs_first_corrected = evecs_first[:, :num_evecs] * evecs_first_signs_i[k]
        evecs_second_corrected = evecs_second[:, :num_evecs] * evecs_second_signs_i[k]
        
        evecs_first_zo = torch.cat((evecs_first_corrected, evecs_first[:, num_evecs:]), dim=1)
        evecs_second_zo = torch.cat((evecs_second_corrected, evecs_second[:, num_evecs:]), dim=1)
        
        
        Cyx_est_k = C_yx_est_i[k][0].cpu()
        
        if apply_zoomout:
        
            Cyx_zo_k = zoomout_custom.zoomout(
                FM_12=Cyx_est_k.to(device), 
                evects1=evecs_second_zo.to(device),
                evects2=evecs_first_zo.to(device),
                nit=evecs_first.shape[1] - num_evecs, step=1,
                A2=template_shape['mass'].to(device)
            )

            p2p_est_k = fmap_util.fmap2pointmap(
                C12=Cyx_zo_k.to(device),
                evecs_x=evecs_second_zo.to(device),
                evecs_y=evecs_first_zo.to(device),
                ).cpu()

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


def get_geo_err_full(dist_mat, corr_first, corr_second, p2p_est_list, p2p_dirichlet, p2p_median, log_file_name):
    geo_err_est_list = []
    
    log_file = open(log_file_name, 'a', buffering=1)

    for i in range(p2p_est_list.shape[0]):
        geo_err_i = geodist_metric.calculate_geodesic_error(
            dist_mat, corr_first.cpu(), corr_second.cpu(), p2p_est_list[i], return_mean=True
        )
        geo_err_est_list.append(geo_err_i * 100)
        
    geo_err_est_list = torch.tensor(geo_err_est_list)

    geo_err_dirichlet = geodist_metric.calculate_geodesic_error(
        dist_mat, corr_first.cpu(), corr_second.cpu(), p2p_dirichlet, return_mean=True
    ) * 100

    geo_err_median = geodist_metric.calculate_geodesic_error(
        dist_mat, corr_first.cpu(), corr_second.cpu(), p2p_median, return_mean=True
    ) * 100

    # print(geo_err_est_list)

    # print(f'Geo error est mean: {geo_err_est_list.mean().item():.3f}')
    # print(f"Geo error est median: {geo_err_est_list.median().item():.3f}")
    # print(f'Geo error dirichlet: {geo_err_dirichlet:.3f}')
    # print(f'Geo error median: {geo_err_median:.3f}')
    
    # log to file the code above, end with \n
    
    log_file.write(f'Pairwise stage \n')
    log_file.write(f'geo_err_est_list: {geo_err_est_list}\n')
    log_file.write(f'Geo error est mean: {geo_err_est_list.mean().item():.3f}\n')
    log_file.write(f"Geo error est median: {geo_err_est_list.median().item():.3f}\n")
    log_file.write(f'Geo error dirichlet: {geo_err_dirichlet:.3f}\n')
    log_file.write(f'Geo error median: {geo_err_median:.3f}\n')
    log_file.write('\n')
    
    
    log_file.close()
    
    return geo_err_est_list.mean().item(), geo_err_est_list.median().item(), geo_err_dirichlet, geo_err_median


def parse_args():
    
    args = argparse.ArgumentParser()
    
    args.add_argument('--experiment_name', type=str, required=True)
    args.add_argument('--checkpoint_name', type=str, required=True)
    
    args.add_argument('--dataset_name', type=str, required=True)
    args.add_argument('--split', type=str, required=True)
    
    args.add_argument('--num_iters_avg', type=int, required=True)
    args.add_argument('--num_samples_median', type=int, required=True)
    args.add_argument('--confidence_threshold', type=float, required=True)
    
    args.add_argument('--num_iters_dataset', type=int, required=True)
    
    args.add_argument('--log_subdir', type=str, required=True)
    
    return args.parse_args()

# class Arguments:
#     def __init__(self):
#         self.experiment_name='partial_0.8_5k_32_2_lambda_0.01_xy'
#         self.checkpoint_name='epoch_99'
        
#         self.dataset_name='SHREC16_cuts_pair'
#         self.split='test'
        
#         self.num_iters_avg=32
#         self.num_samples_median=4
#         self.confidence_threshold=0.3

# run the code with arguments above
# python /home/s94zalek_hpc/shape_matching/notebooks/02.10.2024/test_partial_on_train_data.py --experiment_name partial_0.8_5k_32_2_lambda_0.01_xy --checkpoint_name epoch_99 --dataset_name training_data --split test --num_iters_avg 32 --num_samples_median 4 --confidence_threshold 0.3 --num_iters_dataset 10 --log_subdir test_partial_on_train_data
    

class MetricsContainer:
    
    # make a class that will store the metrics.
    # a new metric is added when the class is indexed with a new key
    # the value is a list of values
    # the class has a method to compute the mean of the values
    
    def __init__(self):
            
        self.metrics = {}
        
    def add_metric(self, key, value):
        
        if key not in self.metrics:
            self.metrics[key] = []
            
        self.metrics[key].append(value)
        
    def compute_mean(self):
            
        mean_metrics = {}
        
        for key in self.metrics:
            
            value = torch.tensor(self.metrics[key])
            mean_metrics[key] = torch.mean(value).item()
            
        return mean_metrics


def run():
    
    # set all random seeds
    torch.manual_seed(0)
    np.random.seed(0)
        
        
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

    num_evecs = config["model_params"]["sample_size"]

    ##########################################
    # Template
    ##########################################

    template_shape = template_dataset.get_template(
        num_evecs=200,
        centering='bbox',
        template_path=f'/home/s94zalek_hpc/shape_matching/data/SURREAL_full/template/{config["sign_net"]["template_type"]}/template.off',
        template_corr=np.loadtxt(
            f'/home/s94zalek_hpc/shape_matching/data/SURREAL_full/template/{config["sign_net"]["template_type"]}/corr.txt',
            dtype=np.int32) - 1
        )    

    ##########################################
    # Logging
    ##########################################

    test_name = 'no_smoothing'

    log_dir = f'{exp_base_folder}/eval/{checkpoint_name}/train_data/{test_name}'
    os.makedirs(log_dir, exist_ok=True)

    fig_dir = f'{log_dir}/figs'
    os.makedirs(fig_dir, exist_ok=True)

    log_file_name = f'{log_dir}/log_{test_name}.txt'


    from my_code.datasets.surreal_dataset_3dc import TemplateSurrealDataset3DC


    augmentations = {
        "remesh": {
                "isotropic": {
                    "n_remesh_iters": 10,
                    "remesh_targetlen": 1,
                    "simplify_strength_min": 0.2,
                    "simplify_strength_max": 0.8,
                },
                "partial": {
                    "probability": 1,
                    "n_remesh_iters": 10,
                    "fraction_to_keep_min": 0.5,
                    "fraction_to_keep_max": 0.9,
                    "n_seed_samples": [1, 5, 25],
                    "weighted_by": "area",
                },
            },
        }

        

    test_dataset = TemplateSurrealDataset3DC(
        shape_path='/lustre/mlnvme/data/s94zalek_hpc-shape_matching/mmap_datas_surreal_train.pth',
        num_evecs=128,
        cache_lb_dir=None,
        return_evecs=True,
        return_fmap=False,
        mmap=True,
        augmentations=augmentations,
        template_path=f'/home/s94zalek_hpc/shape_matching/data/SURREAL_full/template/remeshed/template.off',
        template_corr=np.loadtxt(
            f'/home/s94zalek_hpc/shape_matching/data/SURREAL_full/template/remeshed/corr.txt',
            dtype=np.int32) - 1
    )  
    
    
    metrics_container = MetricsContainer() 
    
    
    for j in tqdm(range(args.num_iters_dataset)):
    
    
        data = test_dataset[j]        

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

        # second mesh


        if config["fmap_direction"] == 'xy':

            Cxy_second_list, evecs_first_signs_list_second, evecs_second_signs_list_second = get_fmaps_evec_signs(
                data['second'], model,
                noise_scheduler, config, args,
                data['first'], sign_corr_net
            )
            # transpose the functional maps
            Cyx_second_list = Cxy_second_list.transpose(2, 3)
            
        else:
            
            Cyx_second_list, evecs_first_signs_list_second, evecs_second_signs_list_second = get_fmaps_evec_signs(
                data['second'], model,
                noise_scheduler, config, args,
                data['first'], sign_corr_net
            )
            
            Cxy_second_list = Cyx_second_list.transpose(2, 3)
            
            
        p2p_est_second, p2p_dirichlet_second, p2p_median_second, confidence_scores_second, dist_y = get_p2p_maps_template(
            data['second'],
            Cyx_second_list, evecs_first_signs_list_second, evecs_second_signs_list_second,
            data['first'], args, log_file_name, config,
            apply_zoomout=False,
        )

        p2p_est_second_rev, p2p_dirichlet_second_rev, p2p_median_second_rev, confidence_scores_second_rev, dist_x = get_p2p_maps_template(
            data['first'],
            Cxy_second_list, evecs_second_signs_list_second, evecs_first_signs_list_second,
            data['second'], args, log_file_name, config,
            apply_zoomout=False,
        )

        # print('##############################')
        # print('# Reverse')
        # print('##############################')
        
        geo_err_est_rev_mean, geo_err_est_rev_median, geo_err_dirichlet_rev, geo_err_median_rev = get_geo_err_full(dist_x, corr_first, corr_second, p2p_est_second_rev, p2p_dirichlet_second_rev, p2p_median_second_rev, log_file_name)
        # print('##############################')
        # print('# Forward')
        # print('##############################')

        geo_err_est_fw_mean, geo_err_est_fw_median, geo_err_dirichlet_fw, geo_err_median_fw = get_geo_err_full(dist_y, corr_second, corr_first, p2p_est_second, p2p_dirichlet_second, p2p_median_second, log_file_name)


        metrics_container.add_metric('geo_err_est_rev_mean', geo_err_est_rev_mean)
        metrics_container.add_metric('geo_err_est_rev_median', geo_err_est_rev_median)
        metrics_container.add_metric('geo_err_dirichlet_rev', geo_err_dirichlet_rev)
        metrics_container.add_metric('geo_err_median_rev', geo_err_median_rev)
        
        metrics_container.add_metric('geo_err_est_fw_mean', geo_err_est_fw_mean)
        metrics_container.add_metric('geo_err_est_fw_median', geo_err_est_fw_median)
        metrics_container.add_metric('geo_err_dirichlet_fw', geo_err_dirichlet_fw)
        metrics_container.add_metric('geo_err_median_fw', geo_err_median_fw)
        
    mean_metrics = metrics_container.compute_mean()
    
    # add arguments to mean_metrics
    mean_metrics_with_args = vars(args)
    mean_metrics_with_args.update(mean_metrics)
    
    
    # log_to_database(mean_metrics, log_file_name)
    
    base_folder = f'/home/s94zalek_hpc/shape_matching/my_code/experiments/ddpm_results/{args.log_subdir}'
    os.makedirs(base_folder, exist_ok=True)

    log_name = f"{mean_metrics_with_args['experiment_name']}_{mean_metrics_with_args['checkpoint_name']}.yaml"

    # save to yaml
    with open(f'{base_folder}/{log_name}', 'w') as f:
        yaml.dump(mean_metrics_with_args, f, sort_keys=False)
    
                
                
if __name__ == '__main__':
    
    run()