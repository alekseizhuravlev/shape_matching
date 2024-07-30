import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
import os
curr_dir = os.getcwd()
if 's94zalek_hpc' in curr_dir:
    user_name = 's94zalek_hpc'
else:
    user_name = 's94zalek'
sys.path.append(f'/home/{user_name}/shape_matching')

import utils.fmap_util as fmap_util
import metrics.geodist_metric as geodist_metric
from my_code.sign_canonicalization.training import predict_sign_change



def count_zero_regions(x_sampled, threshold, percentage):
    incorrect_zero_indices = []
    
    for i in range(x_sampled.shape[0]):
        if (x_sampled[i] > threshold).int().sum() > percentage * x_sampled[i].numel():
            incorrect_zero_indices.append(i)
            
    print(f'Incorrect zero regions: {len(incorrect_zero_indices)} / {x_sampled.shape[0]} = '
          f'{len(incorrect_zero_indices) / x_sampled.shape[0]*100:.2f}%')
    
    return incorrect_zero_indices


def compare_fmap_with_gt(
        Cxy_est, data_x, data_y, sign_corr_net
    ):
    
    # remove the channel dimension if it exists
    if len(Cxy_est.shape) == 3:
        Cxy_est = Cxy_est[0]
        C_gt_xy = data_y['C_gt_xy'][0]
    else:
        C_gt_xy = data_y['C_gt_xy']
        
    # unpack the data  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    verts_first = data_x['verts'].unsqueeze(0).to(device)
    verts_second = data_y['verts'].unsqueeze(0).to(device)
    
    faces_first = data_x['faces'].unsqueeze(0).to(device)
    faces_second = data_y['faces'].unsqueeze(0).to(device)

    evecs_first = data_x['evecs'].unsqueeze(0).to(device)
    evecs_second = data_y['evecs'].unsqueeze(0).to(device)
    
    corr_first = data_x['corr']
    corr_second = data_y['corr']
  
    
    ### get evecs with correct signs
    with torch.no_grad():
        sign_pred_first = predict_sign_change(
            sign_corr_net,
            verts_first, faces_first,
            evecs_first, evecs_cond=None,
            input_type=sign_corr_net.input_type
            )[0]
        sign_pred_second = predict_sign_change(
            sign_corr_net,
            verts_second, faces_second, 
            evecs_second, evecs_cond=None, 
            input_type=sign_corr_net.input_type
            )[0]

    # GT corrected fmap
    C_gt_xy_corr = torch.linalg.lstsq(
        evecs_second.cpu()[0, corr_second] * torch.sign(sign_pred_second).cpu(),
        evecs_first.cpu()[0, corr_first] * torch.sign(sign_pred_first).cpu()
        ).solution 
    
    # sign corrected evecs
    evecs_first_corr = evecs_first.cpu()[0] * torch.sign(sign_pred_first).cpu()
    evecs_second_corr = evecs_second.cpu()[0] * torch.sign(sign_pred_second).cpu()   
    
    
    # hard correspondence 
    p2p_gt = fmap_util.fmap2pointmap(
        C12=C_gt_xy,
        evecs_x=data_x['evecs'],
        evecs_y=data_y['evecs'],
        )
    p2p_corr_gt = fmap_util.fmap2pointmap(
        C12=C_gt_xy_corr,
        evecs_x=evecs_first_corr,
        evecs_y=evecs_second_corr,
        )
    p2p_est = fmap_util.fmap2pointmap(
        Cxy_est,
        evecs_x=evecs_first_corr,
        evecs_y=evecs_second_corr,
        )
    
    # distance matrices
    dist_x = torch.cdist(data_x['verts'], data_x['verts'])
    dist_y = torch.cdist(data_y['verts'], data_y['verts'])

    # geodesic error
    geo_err_gt = geodist_metric.calculate_geodesic_error(dist_x, data_x['corr'], data_y['corr'], p2p_gt, return_mean=False)  
    geo_err_corr_gt = geodist_metric.calculate_geodesic_error(dist_x, data_x['corr'], data_y['corr'], p2p_corr_gt, return_mean=False)
    geo_err_est = geodist_metric.calculate_geodesic_error(dist_x, data_x['corr'], data_y['corr'], p2p_est, return_mean=False)
    
    # mse between sampled and corrected fmap
    mse_fmap = torch.nn.functional.mse_loss(Cxy_est, C_gt_xy_corr)
    
    
    return geo_err_gt, geo_err_corr_gt, geo_err_est, mse_fmap



def calculate_metrics(x_sampled, test_dataset, sign_corr_net):
    
    sign_corr_net.cache_dir = test_dataset.lb_cache_dir
    
    metrics_raw = {
        'geo_err_est': [],
        'geo_err_est_mean': [],
        'mse_fmap': [],
        'geo_err_ratio': [],
        'geo_err_ratio_corr': []
    }

    for i in tqdm(range(len(test_dataset)), desc='Calculating metrics...'):

        batch_i = test_dataset[i]
        fmap_i = x_sampled[i]
        
        # geodesic errors
        geo_err_gt, geo_err_corr_gt, geo_err_est, mse_fmap = compare_fmap_with_gt(
            Cxy_est=fmap_i,
            data_x=batch_i['first'],
            data_y=batch_i['second'],
            sign_corr_net=sign_corr_net
        )

        # geodesic error of predicted fmap
        metrics_raw['geo_err_est'].append(geo_err_est.abs())
        metrics_raw['geo_err_est_mean'].append(geo_err_est.abs().mean())
        
        # mse between predicted and gt fmap
        metrics_raw['mse_fmap'].append(mse_fmap)
        
        # geodesic error ratio to gt fmap and corrected gt fmap
        metrics_raw['geo_err_ratio'].append(geo_err_est.abs().mean() / geo_err_gt.abs().mean())
        metrics_raw['geo_err_ratio_corr'].append(geo_err_est.abs().mean() / geo_err_corr_gt.abs().mean())


    # concatenate the lists
    metrics_raw['geo_err_est'] = torch.cat(metrics_raw['geo_err_est'], dim=0)
    
    for name in ['geo_err_est_mean', 'mse_fmap', 'geo_err_ratio','geo_err_ratio_corr']:
        metrics_raw[name] = torch.stack(metrics_raw[name])
        
    return calculate_statistics(metrics_raw)
    
    


def plot_pck(auc, pcks, thresholds):

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    thresholds = np.linspace(0., 0.1, 40)
    ax.plot(thresholds, pcks, 'r-',
            label=f'auc: {auc:.3f}')
    ax.set_xlim(0., 0.1)
    ax.set_ylim(0, 1)
    ax.set_xscale('linear')
    ax.set_xticks([0.025, 0.05, 0.075, 0.1])
    ax.grid()
    ax.legend()
    ax.set_title('PCK')
    return fig


def calculate_statistics(metrics_raw):
    
    # calculate auc pck
    auc, pcks, thresholds = geodist_metric.plot_pck(
        metrics_raw['geo_err_est'].numpy(), threshold=0.1, steps=40
    )
    fig_pck = plot_pck(auc, pcks, thresholds)
    
    
    # preprocess the metrics
    metrics_payload = {}
    metrics_payload['auc'] = round(auc, 3)
    
    # geodesic error
    metrics_payload['geo_err_mean'] = round(metrics_raw['geo_err_est'].mean().item() * 100, 1)
    
    # geodesic error by shape
    metrics_payload['geo_err_meanByShape_median'] = round(metrics_raw['geo_err_est_mean'].median().item() * 100, 1)
    metrics_payload['geo_err_meanByShape_max'] = round(metrics_raw['geo_err_est_mean'].max().item() * 100, 1)
    metrics_payload['geo_err_meanByShape_min'] = round(metrics_raw['geo_err_est_mean'].min().item() * 100, 1)
    
    # geodesic error ratio to gt fmap
    metrics_payload['geo_err_ratio_mean'] = round(metrics_raw['geo_err_ratio'].mean().item(), 2)
    metrics_payload['geo_err_ratio_median'] = round(metrics_raw['geo_err_ratio'].median().item(), 2)
    metrics_payload['geo_err_ratio_max'] = round(metrics_raw['geo_err_ratio'].max().item(), 2)
    metrics_payload['geo_err_ratio_min'] = round(metrics_raw['geo_err_ratio'].min().item(), 2)
    
    # geodesic error ratio to corrected gt fmap
    metrics_payload['geo_err_ratio_corr_mean'] = round(metrics_raw['geo_err_ratio_corr'].mean().item(), 2)
    metrics_payload['geo_err_ratio_corr_median'] = round(metrics_raw['geo_err_ratio_corr'].median().item(), 2)
    metrics_payload['geo_err_ratio_corr_max'] = round(metrics_raw['geo_err_ratio_corr'].max().item(), 2)
    metrics_payload['geo_err_ratio_corr_min'] = round(metrics_raw['geo_err_ratio_corr'].min().item(), 2)
    
    # mse to gt fmap
    metrics_payload['mse_mean'] = round(metrics_raw['mse_fmap'].mean().item(), 2)
    metrics_payload['mse_median'] = round(metrics_raw['mse_fmap'].median().item(), 2)
    metrics_payload['mse_max'] = round(metrics_raw['mse_fmap'].max().item(), 2)
    metrics_payload['mse_min'] = round(metrics_raw['mse_fmap'].min().item(), 2)
        
    return metrics_payload, fig_pck

