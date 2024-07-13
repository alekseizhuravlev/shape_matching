import torch
import numpy as np
import matplotlib.pyplot as plt

import trimesh
import my_code.diffusion_training.data_loading as data_loading

import os
import shutil
import utils.geometry_util as geometry_util
import utils.shape_util as shape_util
from tqdm import tqdm


if __name__ == '__main__':
    
    dataset_name = 'FAUST_orig'
    
    n_shapes = 1000
    lapl_type = 'mesh'

    split = 'train'
    
    rot_x=180
    rot_y=180
    rot_z=180
    
    along_normal=True
    std=0.01
    noise_clip_low = -0.05
    noise_clip_high = 0.05
    
    scale_min=0.9
    scale_max=1.1
    
    save_folder = f'{dataset_name}_{split}_' +\
                f'rot_{rot_x:.0f}_{rot_y:.0f}_{rot_z:.0f}_' + \
                f'normal_{along_normal}_' + \
                f'noise_{std}_{noise_clip_low}_{noise_clip_high}_' + \
                f'lapl_{lapl_type}_' + \
                f'scale_{scale_min}_{scale_max}'
    
    
    
    
    # get the source dataset
    train_dataset = data_loading.get_val_dataset(
        dataset_name, split, 200, canonicalize_fmap=None
        )[1]    
    
    # prepare the folders
    mesh_folder = f'/home/s94zalek_hpc/shape_matching/data_sign_training/{split}/{save_folder}/meshes'
    diff_folder = f'/home/s94zalek_hpc/shape_matching/data_sign_training/{split}/{save_folder}/diffusion'

    # shutil.rmtree(
    #     f'/home/s94zalek_hpc/shape_matching/data_sign_training/{split}/{save_folder}',
    #     ignore_errors=True
    #     )
    os.makedirs(mesh_folder)
    os.makedirs(diff_folder)


    iterator = tqdm(range(n_shapes))
    
    for epoch in range(n_shapes // len(train_dataset)):
        for i in range(len(train_dataset)):
        
            # get the vertices and faces            
            verts = train_dataset[i]['second']['verts']
            faces = train_dataset[i]['second']['faces']
            
            # augment the vertices
            verts_aug = geometry_util.data_augmentation(verts.unsqueeze(0),
                                                        faces.unsqueeze(0),
                                                        rot_x=rot_x,
                                                        rot_y=rot_y,
                                                        rot_z=rot_z,
                                                        along_normal=along_normal,
                                                        std=std,
                                                        noise_clip_low=noise_clip_low,
                                                        noise_clip_high=noise_clip_high,
                                                        scale_min=scale_min,
                                                        scale_max=scale_max,
                                                        )[0]

            
            # get current iteration
            current_iteration = epoch * len(train_dataset) + i
            
            # save the mesh
            shape_util.write_off(
                f'{mesh_folder}/{current_iteration:04}.off',
                verts_aug.cpu().numpy(),
                faces.cpu().numpy()
                )
            
            
            # read the mesh again
            verts_aug, faces = shape_util.read_shape(f'{mesh_folder}/{current_iteration:04}.off')
            verts_aug = torch.tensor(verts_aug, dtype=torch.float32)
            faces = torch.tensor(faces, dtype=torch.int32)
        
            # calculate and cache the laplacian
            if lapl_type == 'pcl':
                _, _, _, _, evecs_orig, _, _ = geometry_util.get_operators(
                    verts_aug, None,
                    k=128, cache_dir=diff_folder) 
            else:               
                _, _, _, _, evecs_orig, _, _ = geometry_util.get_operators(
                    verts_aug, faces,
                    k=128, cache_dir=diff_folder)

            # update the iterator
            iterator.update(1)
        
