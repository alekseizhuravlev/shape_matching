import os
import numpy as np
import shutil

import torch
from tqdm import tqdm
import argparse

import sys
sys.path.append('/home/s94zalek/shape_matching')

from my_code.datasets.surreal_dataset import SingleSurrealDataset


def save_train_dataset(
        dataset,
        train_indices,
        dataset_folder,
        start_idx,
        end_idx,
    ):
    
    train_folder = f'{dataset_folder}/train'
    os.makedirs(train_folder, exist_ok=True)

    evals_file = os.path.join(train_folder, f'evals_{start_idx}_{end_idx}.txt')
    fmaps_file = os.path.join(train_folder, f'C_gt_xy_{start_idx}_{end_idx}.txt')
    
    print(f'Saving evals to {evals_file}', f'fmaps to {fmaps_file}')
    
    for i in tqdm(train_indices, desc='Saving train dataset'):
        data = dataset[i]
        
        with open(fmaps_file, 'ab') as f:
            np.savetxt(f, data['second']['C_gt_xy'].numpy().flatten().astype(np.float32), newline=" ")
            f.write(b'\n')
            
        with open(evals_file, 'ab') as f:
            np.savetxt(f, data['second']['evals'].numpy().astype(np.float32), newline=" ")
            f.write(b'\n')


def save_first_second_shape_full(dataset, shape_name, indices, base_folder):
    # create a dictionary to store the elements of the dataset
    data_dict = dict()
    for key in dataset[0][shape_name].keys():
        if key != 'L':
            data_dict[key] = []
            
    # copy the data to the dictionary
    for i in tqdm(indices, desc=f'Saving test shape {shape_name}'):
        data = dataset[i][shape_name]
        for key in data.keys():
            if key != 'L':
                if isinstance(data[key], torch.Tensor):
                    data_dict[key].append(data[key].numpy())
                else:
                    raise ValueError(f'Unknown data type for key {key}: {type(data[key])}')
                    data_dict[key].append(data[key])
            
    # create the folder to store the data        
    print('Saving data to disk...')
    folder_shape = f'{base_folder}/{shape_name}'
    os.makedirs(folder_shape)
    
    # save the data to the folder
    for key in data_dict.keys():
        data_dict[key] = np.array(data_dict[key])
        np.save(f'{folder_shape}/{key}.npy', data_dict[key])


def save_test_dataset(
        dataset,
        test_indices,
        dataset_folder,
    ):
    
    # create the test folder
    test_folder = f'{dataset_folder}/test'
    os.makedirs(test_folder, exist_ok=True)

    # save the first shape
    save_first_second_shape_full(
        dataset=dataset,
        shape_name='first', 
        indices=[0],
        base_folder=test_folder
    )    
    # save the second shape
    save_first_second_shape_full(
        dataset=dataset,
        shape_name='second', 
        indices=test_indices,
        base_folder=test_folder
    )
    
    
def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--current_worker', type=int, default=0)
    parser.add_argument('--save_test_set', action='store_true', default=False)
    
    
    parser.add_argument('--n_body_types_male', type=int, default=256)
    parser.add_argument('--n_body_types_female', type=int, default=256)
    parser.add_argument('--n_poses_straight', type=int, default=462)
    parser.add_argument('--n_poses_bent', type=int, default=50)
    parser.add_argument('--num_evecs', type=int, default=32)
    parser.add_argument('--use_same_poses_male_female', action='store_true', default=False)
    parser.add_argument('--train_fraction', type=float, default=0.93)
    
    args = parser.parse_args()
    return args
         
         
if __name__ == '__main__':
    
    args = parse_args()

    n_body_types_male=args.n_body_types_male
    n_body_types_female=args.n_body_types_female
    n_poses_straight=args.n_poses_straight
    n_poses_bent=args.n_poses_bent
    num_evecs=args.num_evecs
    use_same_poses_male_female=args.use_same_poses_male_female
    
    train_fraction = args.train_fraction
    
    np.random.seed(120)
    
    
    ####################################################
    # Dataset
    ####################################################
    
    # create the dataset
    dataset = SingleSurrealDataset(
        n_body_types_male=n_body_types_male,
        n_body_types_female=n_body_types_female,
        n_poses_straight=n_poses_straight,
        n_poses_bent=n_poses_bent,
        num_evecs=num_evecs,
        use_same_poses_male_female=use_same_poses_male_female,
        use_cuda=False
    )    
    
    # sample train/test indices
    train_indices = np.random.choice(len(dataset), int(train_fraction * len(dataset)), replace=False)
    test_indices = np.array([i for i in range(len(dataset)) if i not in train_indices])
    print(f'Number of training samples: {len(train_indices)}, number of test samples: {len(test_indices)}')
    
    # folder to store the dataset
    dataset_name = f'dataset_{n_body_types_male}_{n_body_types_female}_{n_poses_straight}_{n_poses_bent}_{num_evecs}_{int(train_fraction*100)}_samePoses_{int(use_same_poses_male_female)}'
    dataset_folder = f'/home/s94zalek/shape_matching/data/SURREAL_full/full_datasets/{dataset_name}'
    # shutil.rmtree(dataset_folder, ignore_errors=True)
    os.makedirs(dataset_folder, exist_ok=True)
    
    
    ####################################################
    # Saving
    ####################################################


    if args.save_test_set:
        print(f"Saving test dataset...")    
        save_test_dataset(
            dataset=dataset,
            test_indices=test_indices,
            dataset_folder=dataset_folder,
        )
    else:
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
