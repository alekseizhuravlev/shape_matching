import os
import numpy as np
import shutil

import torch
from tqdm import tqdm

import sys
sys.path.append('/home/s94zalek/shape_matching')

from datasets_code.surreal_full import SingleSurrealDataset



def save_train_dataset(
        n_body_types_male,
        n_body_types_female,
        n_poses_straight,
        n_poses_bent,
        num_evecs,
    ):

    fmap_path = '/home/s94zalek/shape_matching/data/SURREAL_full/fmaps'
    evals_path = '/home/s94zalek/shape_matching/data/SURREAL_full/evals'

    os.makedirs(fmap_path, exist_ok=True)
    os.makedirs(evals_path, exist_ok=True)

    evals_file =os.path.join(
        evals_path,
        f'evals_{n_body_types_male}_{n_body_types_female}_{n_poses_straight}_{n_poses_bent}_{num_evecs}.txt'
    )
    fmaps_file =os.path.join(
        fmap_path,
        f'fmaps_{n_body_types_male}_{n_body_types_female}_{n_poses_straight}_{n_poses_bent}_{num_evecs}.txt'
    )
    
    # remove files if they exist
    if os.path.exists(evals_file):
        os.remove(evals_file)
    if os.path.exists(fmaps_file):
        os.remove(fmaps_file)
        
    print(f'Saving evals to {evals_file}', f'fmaps to {fmaps_file}')
    
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        
        with open(fmaps_file, 'ab') as f:
            np.savetxt(f, data['second']['C_gt_xy'].numpy().flatten().astype(np.float32), newline=" ")
            f.write(b'\n')
            
        with open(evals_file, 'ab') as f:
            np.savetxt(f, data['second']['evals'].numpy().astype(np.float32), newline=" ")
            f.write(b'\n')


def save_single_shape_full(dataset, shape_name, n_shapes, base_folder):
    # create a dictionary to store the elements of the dataset
    data_dict = dict()
    for key in dataset[0][shape_name].keys():
        if key != 'L':
            data_dict[key] = []
            
    # copy the data to the dictionary
    for i in tqdm(range(n_shapes)):
        data = dataset[i][shape_name]
        for key in data.keys():
            if key != 'L':
                if isinstance(data[key], torch.Tensor):
                    data_dict[key].append(data[key].numpy())
                else:
                    data_dict[key].append(data[key])
            
    # create the folder to store the data        
    print('Saving data')
    folder_shape = f'{base_folder}/{shape_name}'
    os.makedirs(folder_shape)
    
    # save the data to the folder
    for key in data_dict.keys():
        data_dict[key] = np.array(data_dict[key])
        np.save(f'{folder_shape}/{key}.npy', data_dict[key])


def save_test_dataset(
        n_body_types_male,
        n_body_types_female,
        n_poses_straight,
        n_poses_bent,
        num_evecs,
    ):
    # create the dataset
    dataset = SingleSurrealDataset(
        n_body_types_male=n_body_types_male,
        n_body_types_female=n_body_types_female,
        n_poses_straight=n_poses_straight,
        n_poses_bent=n_poses_bent,
        num_evecs=num_evecs
    )
    # folder to store the dataset
    dataset_name = f'dataset_{n_body_types_male}_{n_body_types_female}_{n_poses_straight}_{n_poses_bent}_{num_evecs}'
    base_folder = f'/home/s94zalek/shape_matching/data/SURREAL_full/full_datasets/{dataset_name}'
    shutil.rmtree(base_folder, ignore_errors=True)

    # save the first shape
    save_single_shape_full(dataset=dataset,
            shape_name='first', 
            n_shapes=1,
            base_folder=base_folder)
    
    # save the second shape
    save_single_shape_full(dataset=dataset,
            shape_name='second', 
            n_shapes=len(dataset),
            base_folder=base_folder)
         


if __name__ == '__main__':
    
    dataset = SingleSurrealDataset(
        n_body_types_male=n_body_types_male,
        n_body_types_female=n_body_types_female,
        n_poses_straight=n_poses_straight,
        n_poses_bent=n_poses_bent,
        num_evecs=num_evecs
    )
    
    
        
    save_evals_fmaps(
        n_body_types_male=160,
        n_body_types_female=160,
        n_poses_straight=320,
        n_poses_bent=0,
        num_evecs=32,
    )
    
    # 4096 shapes
    save_full_dataset(
        n_body_types_male=32,
        n_body_types_female=32,
        n_poses_straight=64,
        n_poses_bent=0,
        num_evecs=32,
    )
    
    # 1024 shapes
    # save_full_dataset(
    #     n_body_types_male=16,
    #     n_body_types_female=16,
    #     n_poses_straight=32,
    #     n_poses_bent=0,
    #     num_evecs=32,
    # )
        