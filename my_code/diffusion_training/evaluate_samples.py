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
    Cxy_est, Cxy_gt,
    evecs_x, evecs_y,
    evecs_trans_x, evecs_trans_y,
    verts_x, verts_y,
    corr_x, corr_y
    ):
    
    # hard correspondence 
    p2p_gt = fmap_util.fmap2pointmap(
        Cxy_gt,
        evecs_x,
        evecs_y,
        )
    p2p_est = fmap_util.fmap2pointmap(
        Cxy_est,
        evecs_x,
        evecs_y,
        )
    
    # soft correspondence
    Pyx_gt = evecs_y @ Cxy_gt @ evecs_trans_x
    Pyx_est = evecs_y @ Cxy_est @ evecs_trans_x
    
    # distance matrices
    dist_x = torch.cdist(verts_x, verts_x)
    dist_y = torch.cdist(verts_y, verts_y)

    # geodesic error
    geo_err_gt = geodist_metric.calculate_geodesic_error(dist_x, corr_x, corr_y, p2p_gt, return_mean=False)    
    geo_err_est = geodist_metric.calculate_geodesic_error(dist_x, corr_x, corr_y, p2p_est, return_mean=False)
    
    # print results
    # print(f'Cxy_est - Cxy_gt: {torch.sum(torch.abs(Cxy_est - Cxy_gt)):.2f}')
    # print(f'Pyx_est - Pyx_gt: {torch.sum(torch.abs(Pyx_est - Pyx_gt)):.2f}')
    # print(f'Geodesic error GT: {geo_err_gt.abs().sum():.2f}, EST: {geo_err_est.abs().sum():.2f}')
    
    return geo_err_gt, geo_err_est
    