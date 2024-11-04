import numpy as np
import my_code.diffusion_training_sign_corr.data_loading as data_loading
import yaml
from tqdm import tqdm
import metrics.geodist_metric as geodist_metric
from utils.shape_util import compute_geodesic_distmat
import torch


def get_geo_err_3dcoded(dataset_name):

    single_dataset, pair_dataset = data_loading.get_val_dataset(
        dataset_name, 'test', 128, preload=False, return_evecs=True, centering='bbox'
    )

    dist_mat_list = []

    for i in tqdm(range(len(single_dataset)), desc='Computing dist mats'):

        data_i = single_dataset[i]

        dist_mat = torch.tensor(
            compute_geodesic_distmat(data_i['verts'].numpy(), data_i['faces'].numpy())    
        )
        
        dist_mat_list.append(dist_mat)

    path_3dc = f'/lustre/mlnvme/data/s94zalek_hpc-shape_matching/results_baselines/3D_CODED/{dataset_name}'

    geo_err_list = []

    for i in tqdm(range(len(pair_dataset)), desc='Computing geo err'):
        
        data_i = pair_dataset[i]
        
        first_idx = data_i['first']['id']
        second_idx = data_i['second']['id']
        
        p2p_3dc = torch.tensor(
            np.loadtxt(f'{path_3dc}/{first_idx}-{second_idx}.txt')
        ).int()
        
        dist_x = dist_mat_list[first_idx]
        dist_y = dist_mat_list[second_idx]
        
        corr_first = data_i['first']['corr']
        corr_second = data_i['second']['corr']
        
        # geo_err = geodist_metric.calculate_geodesic_error(
        #     dist_x, corr_first.cpu(), corr_second.cpu(), p2p_3dc, return_mean=True
        # ) * 100
        
        geo_err = geodist_metric.calculate_geodesic_error(
            dist_y, corr_second.cpu(), corr_first.cpu(), p2p_3dc, return_mean=True
        ) * 100
        
        geo_err_list.append(geo_err)
        
    geo_err_list = torch.tensor(geo_err_list)
    print(f'{dataset_name}, mean geo err: {geo_err_list.mean()}, median: {geo_err_list.median()}')
    
    
    
def get_geo_err_consistent(dataset_name):

    single_dataset, pair_dataset = data_loading.get_val_dataset(
        dataset_name, 'test', 128, preload=False, return_evecs=True, centering='bbox'
    )

    dist_mat_list = []

    for i in tqdm(range(len(single_dataset)), desc='Computing dist mats'):

        data_i = single_dataset[i]

        dist_mat = torch.tensor(
            compute_geodesic_distmat(data_i['verts'].numpy(), data_i['faces'].numpy())    
        )
        
        dist_mat_list.append(dist_mat)

    
    path = f'/home/s94zalek_hpc/baselines/Spatially-and-Spectrally-Consistent-Deep-Functional-Maps/data/results/{dataset_name[:-5]}/p2p_21'

    geo_err_list = []

    for i in tqdm(range(len(pair_dataset))):
        
        data_i = pair_dataset[i]
        
        first_idx = data_i['first']['id']
        second_idx = data_i['second']['id']
        
        p2p = torch.tensor(
            np.loadtxt(f'{path}/{first_idx}_{second_idx}.txt')
        ).int()
        
        dist_x = dist_mat_list[first_idx]
        dist_y = dist_mat_list[second_idx]
        
        corr_first = data_i['first']['corr']
        corr_second = data_i['second']['corr']
        
        geo_err = geodist_metric.calculate_geodesic_error(
            dist_x, corr_first.cpu(), corr_second.cpu(), p2p, return_mean=True
        ) * 100
        
        # geo_err = geodist_metric.calculate_geodesic_error(
        #     dist_y, corr_second.cpu(), corr_first.cpu(), p2p, return_mean=True
        # ) * 100
        
        geo_err_list.append(geo_err)
        
        # print(geo_err)
        # break
        
    geo_err_list = torch.tensor(geo_err_list)
    print(f'{dataset_name}, mean geo err: {geo_err_list.mean()}, median: {geo_err_list.median()}')
    
    
if __name__ == '__main__':
    
    dataset_names = [
        # 'FAUST_r_pair',
        # 'FAUST_a_pair', 
        # 'SCAPE_r_pair',
        'SCAPE_a_pair',
        # 'SHREC19_r_pair'
        ]
    
    for dataset_name in dataset_names:
        get_geo_err_consistent(dataset_name)