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

    
from my_code.datasets.surreal_dataset_3dc import TemplateSurrealDataset3DC


def save_train_dataset(
        dataset,
        train_indices,
        dataset_folder,
        start_idx,
        end_idx,
    ):
    
    curr_time = time.time()
    
    train_folder = f'{dataset_folder}/train'
    os.makedirs(train_folder, exist_ok=True)

    evals_file = os.path.join(train_folder, f'evals_{start_idx}_{end_idx}.txt')
    fmaps_file = os.path.join(train_folder, f'C_gt_xy_{start_idx}_{end_idx}.txt')
    
    print(f'Saving evals to {evals_file}', f'fmaps to {fmaps_file}')
    
    for i, idx in enumerate(train_indices):
        data = dataset[idx]
        
        with open(fmaps_file, 'ab') as f:
            np.savetxt(f, data['second']['C_gt_xy'].numpy().flatten().astype(np.float32), newline=" ")
            f.write(b'\n')
            
        with open(evals_file, 'ab') as f:
            np.savetxt(f, data['second']['evals'].numpy().astype(np.float32), newline=" ")
            f.write(b'\n')
            
        if i % 100 == 0:
            time_elapsed = time.time() - curr_time
            print(f'{i}/{len(train_indices)}, time: {time_elapsed:.2f}, avg: {time_elapsed / (i + 1):.2f}')


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_workers', type=int)
    parser.add_argument('--current_worker', type=int)
    
    parser.add_argument('--num_evecs', type=int)
    
    args = parser.parse_args()
    
    # python my_code/datasets/cache_surreal_3dc.py --n_workers 4 --current_worker 0 --num_evecs 32
    
    return args
         
         
if __name__ == '__main__':
    
    args = parse_args()

    np.random.seed(120)
    
    num_evecs = args.num_evecs
        
    ####################################################
    # Dataset
    ####################################################
    
    # create the dataset
    dataset = TemplateSurrealDataset3DC(
        shape_path=f'/home/{user_name}/3D-CODED/data/datas_surreal_train.pth',
        num_evecs=num_evecs,
        use_cuda=False
    )    
    
    # sample train/test indices
    train_indices = list(range(len(dataset)))
    print(f'Number of training samples: {len(train_indices)}')
    
    # folder to store the dataset
    dataset_name = f'dataset_3dc_{num_evecs}'
    dataset_folder = f'/home/{user_name}/shape_matching/data/SURREAL_full/full_datasets/{dataset_name}'
    # shutil.rmtree(dataset_folder, ignore_errors=True)
    os.makedirs(dataset_folder, exist_ok=True)
    
    
    ####################################################
    # Saving
    ####################################################

    n_workers = args.n_workers
    current_worker = args.current_worker
    
    n_samples = len(dataset)
    samples_per_worker = n_samples // n_workers
    start = current_worker * samples_per_worker
    end = (current_worker + 1) * samples_per_worker
    if current_worker == n_workers - 1:
        end = n_samples
        
    print(f'Worker {current_worker} processing samples from {start} to {end}')
    train_indices = train_indices[start:end]
        

    print(f"Saving train dataset...")
    save_train_dataset(
        dataset=dataset,
        train_indices=train_indices,
        dataset_folder=dataset_folder,
        start_idx=start,
        end_idx=end
    )
