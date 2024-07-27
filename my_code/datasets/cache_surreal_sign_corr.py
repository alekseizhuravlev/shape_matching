import os
import numpy as np
import shutil

import torch
from tqdm import tqdm
import argparse
import time

import sys
import os
curr_dir = os.getcwd()
if 's94zalek_hpc' in curr_dir:
    user_name = 's94zalek_hpc'
else:
    user_name = 's94zalek'
sys.path.append(f'/home/{user_name}/shape_matching/')

from my_code.sign_canonicalization.training import predict_sign_change
import networks.diffusion_network as diffusion_network
import my_code.utils.plotting_utils as plotting_utils
import matplotlib.pyplot as plt
    
    
def visualize_before_after(data, C_xy_corr, figures_folder, idx):
        l = 0
        h = 32

        fig, axs = plt.subplots(1, 4, figsize=(15, 5))

        plotting_utils.plot_Cxy(fig, axs[0], data['second']['C_gt_xy'][0],
                                'before', l, h, show_grid=False, show_colorbar=False)
        plotting_utils.plot_Cxy(fig, axs[1], C_xy_corr,
                                'after', l, h, show_grid=False, show_colorbar=False)
        plotting_utils.plot_Cxy(fig, axs[2], data['second']['C_gt_xy'][0] - C_xy_corr,
                        'diff', l, h, show_grid=False, show_colorbar=False)
        plotting_utils.plot_Cxy(fig, axs[3], data['second']['C_gt_xy'][0].abs() - C_xy_corr.abs(),
                        'abs diff', l, h, show_grid=False, show_colorbar=False)

        # save the figure
        fig.savefig(f'{figures_folder}/{idx}.png')
        plt.close(fig)
        
    
def get_corrected_fmap(data, net, net_input_type):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    
    verts_first = data['first']['verts'].unsqueeze(0).to(device)
    verts_second = data['second']['verts'].unsqueeze(0).to(device)
    
    faces_first = data['first']['faces'].unsqueeze(0).to(device)
    faces_second = data['second']['faces'].unsqueeze(0).to(device)

    evecs_first = data['first']['evecs'].unsqueeze(0).to(device)
    evecs_second = data['second']['evecs'].unsqueeze(0).to(device)

    corr_first = data['first']['corr']
    corr_second = data['second']['corr']


    # predict the sign change
    with torch.no_grad():
        sign_pred_first = predict_sign_change(net, verts_first, faces_first, evecs_first, 
                                                evecs_cond=None, input_type=net_input_type)[0]
        sign_pred_second = predict_sign_change(net, verts_second, faces_second, evecs_second, 
                                                evecs_cond=None, input_type=net_input_type)[0]

    C_xy_pred = torch.linalg.lstsq(
        evecs_second.cpu()[0, corr_second] * torch.sign(sign_pred_second).cpu(),
        evecs_first.cpu()[0, corr_first] * torch.sign(sign_pred_first).cpu()
        ).solution
    
    return C_xy_pred
    
    
    

def save_train_dataset(
        dataset,
        train_indices,
        dataset_folder,
        start_idx,
        end_idx,
        **net_params
    ):
    
    curr_time = time.time()
    
    train_folder = f'{dataset_folder}/train'
    os.makedirs(train_folder, exist_ok=True)
    
    # figures_folder = f'{train_folder}/figures'
    # os.makedirs(figures_folder, exist_ok=True)

    evals_file = os.path.join(train_folder, f'evals_{start_idx}_{end_idx}.txt')
    fmaps_file = os.path.join(train_folder, f'C_gt_xy_{start_idx}_{end_idx}.txt')
    
    # remove if exist
    if os.path.exists(evals_file):
        print(f'Removing {evals_file}')
        os.remove(evals_file)
    if os.path.exists(fmaps_file):
        print(f'Removing {fmaps_file}')
        os.remove(fmaps_file)
    
    print(f'Saving evals to {evals_file}', f'fmaps to {fmaps_file}')
    
    for i, idx in enumerate(train_indices):
        data = dataset[idx]
        
        evals = data['second']['evals']
        C_xy_corr = get_corrected_fmap(
            data=data,
            **net_params
        )
                
        with open(fmaps_file, 'ab') as f:
            np.savetxt(f, C_xy_corr.numpy().flatten().astype(np.float32), newline=" ")
            f.write(b'\n')
            
        with open(evals_file, 'ab') as f:
            np.savetxt(f, evals.numpy().astype(np.float32), newline=" ")
            f.write(b'\n')
        
            
        if i % 100 == 0 or i == 25:
            time_elapsed = time.time() - curr_time
            print(f'{i}/{len(train_indices)}, time: {time_elapsed:.2f}, avg: {time_elapsed / (i + 1):.2f}',
                  flush=True)


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_workers', type=int)
    parser.add_argument('--current_worker', type=int)
    
    parser.add_argument('--num_evecs', type=int)
    
    parser.add_argument('--net_path', type=str)
    parser.add_argument('--net_input_type', type=str)
    parser.add_argument('--evecs_per_support', type=int)
    
    args = parser.parse_args()
    
    # python my_code/datasets/cache_surreal_sign_corr.py --n_workers 1 --current_worker 0 --num_evecs 32 --net_input_type wks --evecs_per_support 4 --net_path /home/s94zalek_hpc/shape_matching/my_code/experiments/sign_estimator_no_aug/40000.pth
    
    return args
         
         
if __name__ == '__main__':
    
    args = parse_args()

    np.random.seed(120)
    
    num_evecs = args.num_evecs
        
    ####################################################
    # Dataset
    ####################################################
    
    from my_code.datasets.surreal_dataset_3dc import TemplateSurrealDataset3DC
    
    # create the dataset
    dataset = TemplateSurrealDataset3DC(
        shape_path=f'/home/{user_name}/3D-CODED/data/datas_surreal_train.pth',
        num_evecs=num_evecs,
        use_cuda=False,
        cache_lb_dir=None,
        return_evecs=True
    )    
    print('Dataset created')
    
    # sample train/test indices
    train_indices = list(range(len(dataset)))
    print(f'Number of training samples: {len(train_indices)}')
    
    # folder to store the dataset
    dataset_name = f'dataset_3dc_corrected_noAug_{num_evecs}'
    dataset_folder = f'/home/{user_name}/shape_matching/data/SURREAL_full/full_datasets/{dataset_name}'
    # shutil.rmtree(dataset_folder, ignore_errors=True)
    os.makedirs(dataset_folder, exist_ok=True)
    
    
    ####################################################
    # Sign correction network
    ####################################################

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    net = diffusion_network.DiffusionNet(
        in_channels=num_evecs,
        out_channels=num_evecs // args.evecs_per_support,
        cache_dir=None,
        input_type=args.net_input_type,
        k_eig=128,
        n_block=6
        ).to(device)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.load_state_dict(torch.load(args.net_path, map_location=device))
    
    ####################################################
    # Saving
    ####################################################

    # current and total workers
    n_workers = args.n_workers
    current_worker = args.current_worker
    
    # samples per worker
    n_samples = len(train_indices)
    samples_per_worker = n_samples // n_workers
    
    # start - end indices
    start = current_worker * samples_per_worker
    end = (current_worker + 1) * samples_per_worker
    if current_worker == n_workers - 1:
        end = n_samples
        
    print(f'Worker {current_worker} processing samples from {start} to {end}')
    
    # indices for this worker
    train_indices = train_indices[start:end]
        

    print(f"Saving train dataset...")
    save_train_dataset(
        dataset=dataset,
        train_indices=train_indices,
        dataset_folder=dataset_folder,
        start_idx=start,
        end_idx=end,
        
        # sign corr net parameters
        net=net,
        net_input_type=args.net_input_type
    )
