import os, re
import numpy as np
import scipy.io as sio
from itertools import product
from glob import glob
import trimesh
import shutil

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import sys
sys.path.append('/home/s94zalek/shape_matching')

from utils.geometry_util import get_operators
from datasets_code.generate_surreal_shapes import generate_shapes


def get_spectral_ops(item, num_evecs, cache_dir=None):
    if cache_dir is not None and not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    _, mass, L, evals, evecs, _, _ = get_operators(item['verts'], item.get('faces'),
                                                   k=num_evecs,
                                                   cache_dir=cache_dir)
    evecs_trans = evecs.T * mass[None]
    item['evecs'] = evecs[:, :num_evecs]
    item['evecs_trans'] = evecs_trans[:num_evecs]
    item['evals'] = evals[:num_evecs]
    item['mass'] = mass
    item['L'] = L.to_dense()

    return item


class SingleSurrealDataset(Dataset):
    def __init__(self,
                 n_body_types_male,
                 n_body_types_female,
                 n_poses_straight,
                 n_poses_bent,
                 num_evecs=200
                 ):

        self.data_root = '/home/s94zalek/shape_matching/data/SURREAL'
        self.num_evecs = num_evecs
        
        # generate male and female unbent shapes
        # male_shapes = generate_shapes(n_body_types=n_body_types, n_poses=n_poses, male=True, bent=False)
        # female_shapes = generate_shapes(n_body_types=n_body_types, n_poses=n_poses, male=False, bent=False)
        
        # # concatenate them
        # self.shapes = {
        #     'verts': torch.cat([male_shapes['verts'], female_shapes['verts']], axis=0),
        #     'faces': torch.cat([male_shapes['faces'], female_shapes['faces']], axis=0),
        #     'poses': torch.cat([male_shapes['poses'], female_shapes['poses']], axis=0),
        #     'betas': torch.cat([male_shapes['betas'], female_shapes['betas']], axis=0),
        # }
        
        # if include_bent:
        #     # bent shapes are 1/6 of the total number of shapes, but at least one
        #     n_body_types_bent = max(n_body_types // 6, 1)
        #     n_poses_bent = max(n_poses // 6, 1)
            
        #     # generate male and female bent shapes
        #     male_bent = generate_shapes(
        #         n_body_types=n_body_types_bent, n_poses=n_poses_bent,
        #         male=True, bent=True)
        #     female_bent = generate_shapes(
        #         n_body_types=n_body_types_bent, n_poses=n_poses_bent,
        #         male=False, bent=True)
            
        #     # concatenate them
        #     self.shapes = {
        #         'verts': torch.cat([self.shapes['verts'], male_bent['verts'], female_bent['verts']], axis=0),
        #         'faces': torch.cat([self.shapes['faces'], male_bent['faces'], female_bent['faces']], axis=0),
        #         'poses': torch.cat([self.shapes['poses'], male_bent['poses'], female_bent['poses']], axis=0),
        #         'betas': torch.cat([self.shapes['betas'], male_bent['betas'], female_bent['betas']], axis=0),
        #     }
        
        self.shapes = generate_shapes(n_body_types_male, n_body_types_female, n_poses_straight, n_poses_bent)
            
        
        # load template mesh
        self.template_mesh = trimesh.load('/home/s94zalek/shape_matching/data/SURREAL_full/template/template.ply')

        # sanity check
        assert len(self.shapes) > 0, f'No shapes found'
        assert self.template_mesh is not None, f'No template_mesh found'
        
        # make the template object
        self.template = {
            'name': 'template',
            'verts': torch.tensor(self.template_mesh.vertices).float(),
            'faces': torch.tensor(self.template_mesh.faces).long(),
            'corr': list(range(len(self.template_mesh.vertices))),
        }
        self.template = get_spectral_ops(self.template, num_evecs=self.num_evecs)
     
    
    def get_functional_map(self, data_x, data_y):

        # calculate the map
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        C_gt_xy = torch.linalg.lstsq(
            data_y['evecs'][data_y['corr']].to(device),
            data_x['evecs'][data_x['corr']].to(device)
            ).solution.to('cpu')
        
        C_gt_yx = torch.linalg.lstsq(
            data_x['evecs'][data_x['corr']].to(device),
            data_y['evecs'][data_y['corr']].to(device)
            ).solution.to('cpu')

        return C_gt_xy, C_gt_yx
        
        
    def __getitem__(self, index):
        
        item = dict()
        item['name'] = str(index)
        
        item['verts'] = self.shapes['verts'][index]
        item['faces'] = self.shapes['faces'][index]
        item['poses'] = self.shapes['poses'][index]
        item['betas'] = self.shapes['betas'][index]
        
        # get eigenfunctions/eigenvalues
        item = get_spectral_ops(item, num_evecs=self.num_evecs,) #cache_dir=os.path.join(self.data_root, 'diffusion'))
        
        # 1 to 1 correspondence
        item['corr'] = list(range(len(item['verts'])))        
        
        payload =  {
            'first': self.template,
            'second': item,
        }
        payload['C_gt_xy'], payload['C_gt_yx'] = self.get_functional_map(payload['first'], payload['second'])
        
        return payload


    def __len__(self):
        return len(self.shapes['verts'])


if __name__ == '__main__':
    
    n_body_types_male = 160
    n_body_types_female = 160
    n_poses_straight = 320
    n_poses_bent = 0
    num_evecs = 32
    
    dataset = SingleSurrealDataset(
        n_body_types_male=n_body_types_male,
        n_body_types_female=n_body_types_female,
        n_poses_straight=n_poses_straight,
        n_poses_bent=n_poses_bent,
        num_evecs=num_evecs
    )
    fmap_path = '/home/s94zalek/shape_matching/data/SURREAL_full/fmaps'
    evals_path = '/home/s94zalek/shape_matching/data/SURREAL_full/evals'
    # evecs_path = '/home/s94zalek/shape_matching/data/SURREAL_full/eigenvectors/train'

    os.makedirs(fmap_path, exist_ok=True)
    os.makedirs(evals_path, exist_ok=True)
    # os.makedirs(evecs_path, exist_ok=True)

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
            np.savetxt(f, data['C_gt_xy'].numpy().flatten().astype(np.float32), newline=" ")
            f.write(b'\n')
            
        with open(evals_file, 'ab') as f:
            np.savetxt(f, data['second']['evals'].numpy().astype(np.float32), newline=" ")
            f.write(b'\n')
            
        # with open(os.path.join(evecs_path, f'{int(data["second"]["name"]):05d}.txt'), 'wb') as f:
        #     np.savetxt(f, data['second']['evecs'].numpy())
            
        
        