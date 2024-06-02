import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/s94zalek/shape_matching')

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
        data_y['C_gt_xy'] = data_y['C_gt_xy'][0]
    
    # hard correspondence 
    p2p_gt = fmap_util.fmap2pointmap(
        C12=data_y['C_gt_xy'],
        evecs_x=data_x['evecs'],
        evecs_y=data_y['evecs'],
        )
    p2p_est = fmap_util.fmap2pointmap(
        Cxy_est,
        evecs_x=data_x['evecs'],
        evecs_y=data_y['evecs'],
        )
    
    # soft correspondence
    Pyx_gt = data_y['evecs'] @ data_y['C_gt_xy'] @ data_x['evecs_trans']
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