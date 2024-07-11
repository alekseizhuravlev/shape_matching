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



def count_zero_regions(x_sampled, threshold, percentage):
    incorrect_zero_indices = []
    
    for i in range(x_sampled.shape[0]):
        if (x_sampled[i] > threshold).int().sum() > percentage * x_sampled[i].numel():
            incorrect_zero_indices.append(i)
            
    print(f'Incorrect zero regions: {len(incorrect_zero_indices)} / {x_sampled.shape[0]} = '
          f'{len(incorrect_zero_indices) / x_sampled.shape[0]*100:.2f}%')
    
    return incorrect_zero_indices


def compare_fmap_with_gt(
    Cxy_est, data_x, data_y
    ):
    
    # remove the channel dimension if it exists
    if len(Cxy_est.shape) == 3:
        Cxy_est = Cxy_est[0]
        C_gt_xy = data_y['C_gt_xy'][0]
        # data_y['C_gt_xy'] = data_y['C_gt_xy'][0]
    else:
        C_gt_xy = data_y['C_gt_xy']
    
    # hard correspondence 
    p2p_gt = fmap_util.fmap2pointmap(
        C12=C_gt_xy,
        evecs_x=data_x['evecs'],
        evecs_y=data_y['evecs'],
        )
    p2p_est = fmap_util.fmap2pointmap(
        Cxy_est,
        evecs_x=data_x['evecs'],
        evecs_y=data_y['evecs'],
        )
    
    # soft correspondence
    Pyx_gt = data_y['evecs'] @ C_gt_xy @ data_x['evecs_trans']
    Pyx_est = data_y['evecs'] @ Cxy_est @ data_x['evecs_trans']
    
    # distance matrices
    dist_x = torch.cdist(data_x['verts'], data_x['verts'])
    dist_y = torch.cdist(data_y['verts'], data_y['verts'])

    # geodesic error
    geo_err_gt = geodist_metric.calculate_geodesic_error(dist_x, data_x['corr'], data_y['corr'], p2p_gt, return_mean=False)    
    geo_err_est = geodist_metric.calculate_geodesic_error(dist_x, data_x['corr'], data_y['corr'], p2p_est, return_mean=False)
    
    return geo_err_gt, geo_err_est, p2p_gt, p2p_est



def calculate_metrics(x_sampled, test_dataset):
    
    metrics = {
        'geo_err_gt': [],
        'geo_err_est': [],
        'p2p_gt': [],
        'p2p_est': [],
        'mse_abs': [],
        'geo_err_ratio': [],
        'auc': [],
        'pcks': [],
    }

    for i in tqdm(range(len(test_dataset)), desc='Calculating metrics...'):

        batch_i = test_dataset[i]
        fmap_i = x_sampled[i]
        x_gt_i = batch_i['second']['C_gt_xy']
        
        # MSE of absolute values
        mse_abs = torch.sum((fmap_i.abs() - x_gt_i.abs())**2)

        # p2p error
        geo_err_gt, geo_err_est, p2p_gt, p2p_est = compare_fmap_with_gt(
            Cxy_est=fmap_i,
            data_x=batch_i['first'],
            data_y=batch_i['second']
        )
        
        auc, pcks, thresholds = geodist_metric.plot_pck(
            geo_err_est.numpy(), threshold=0.1, steps=40
        )

        metrics['geo_err_gt'].append(geo_err_gt.abs().mean())
        metrics['geo_err_est'].append(geo_err_est.abs().mean())
        metrics['geo_err_ratio'].append(geo_err_est.abs().mean() / geo_err_gt.abs().mean())
        
        metrics['p2p_gt'].append(p2p_gt)
        metrics['p2p_est'].append(p2p_est)
        
        metrics['mse_abs'].append(mse_abs)
        
        metrics['auc'].append(auc)
        metrics['pcks'].append(pcks)
        
        
    # calculate mean auc and pcks
    # metrics['auc'] = np.mean(metrics['auc'], axis=0)
    # metrics['pcks'] = np.mean(metrics['pcks'], axis=0)
    
    metrics['pcks'] = np.array(metrics['pcks'])
        
    for k in ['geo_err_gt', 'geo_err_est', 'mse_abs', 'geo_err_ratio', 'auc', 'pcks']:
        metrics[k] = torch.tensor(metrics[k])
        
    return metrics


def plot_pck(metrics, title):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    thresholds = np.linspace(0., 0.1, 40)
    ax.plot(thresholds, torch.mean(metrics['pcks'], axis=0), 'r-',
            label=f'auc: {torch.mean(metrics["auc"]):.2f}')
    ax.set_xlim(0., 0.1)
    ax.set_ylim(0, 1)
    ax.set_xscale('linear')
    ax.set_xticks([0.025, 0.05, 0.075, 0.1])
    ax.grid()
    ax.legend()
    ax.set_title(title)
    return fig


def preprocess_metrics(metrics):
    metrics_payload = {}
    
    metrics_payload['auc'] = round(metrics['auc'].mean(dim=0).item(), 2)
    
    # geodesic error
    metrics_payload['geo_err_mean'] = round(metrics['geo_err_est'].mean().item() * 100, 1)
    metrics_payload['geo_err_median'] = round(metrics['geo_err_est'].median().item() * 100, 1)
    
    # geodesic error ratio to gt fmap
    metrics_payload['geo_err_ratio_mean'] = round(metrics['geo_err_ratio'].mean().item(), 2)
    metrics_payload['geo_err_ratio_median'] = round(metrics['geo_err_ratio'].median().item(), 2)
    metrics_payload['geo_err_ratio_max'] = round(metrics['geo_err_ratio'].max().item(), 2)
    metrics_payload['geo_err_ratio_min'] = round(metrics['geo_err_ratio'].min().item(), 2)
    
    # mse to gt fmap
    metrics_payload['mse_mean'] = round(metrics['mse_abs'].mean().item(), 2)
    metrics_payload['mse_median'] = round(metrics['mse_abs'].median().item(), 2)
    metrics_payload['mse_max'] = round(metrics['mse_abs'].max().item(), 2)
    metrics_payload['mse_min'] = round(metrics['mse_abs'].min().item(), 2)
    
    return metrics_payload