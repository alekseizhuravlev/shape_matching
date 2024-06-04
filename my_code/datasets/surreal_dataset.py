import json
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
from my_code.datasets.generate_surreal_shapes import generate_shapes


def get_spectral_ops(item, num_evecs, cache_dir=None):
    if cache_dir is not None and not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    _, mass, L, evals, evecs, _, _ = get_operators(item['verts'], item.get('faces'),
                                                   k=num_evecs,
                                                   cache_dir=cache_dir)
    evals = evals.unsqueeze(0)
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

        self.data_root = '/home/s94zalek/shape_matching/data/SURREAL_full'
        self.num_evecs = num_evecs

        # generate male-female, straight-bent shapes
        self.shapes = generate_shapes(n_body_types_male, n_body_types_female, n_poses_straight, n_poses_bent)
        
        # load template mesh
        self.template_mesh = trimesh.load('/home/s94zalek/shape_matching/data/SURREAL_full/template/template.ply')

        # sanity check
        assert len(self.shapes) > 0, f'No shapes found'
        assert self.template_mesh is not None, f'No template_mesh found'
        
        # make the template object
        self.template = {
            'id': torch.tensor(-1),
            'verts': torch.tensor(self.template_mesh.vertices).float(),
            'faces': torch.tensor(self.template_mesh.faces).long(),
            'corr': torch.tensor(list(range(len(self.template_mesh.vertices)))),
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
        
        # print(index, type(index))
        item['id'] = torch.tensor(index)
        
        
        item['verts'] = self.shapes['verts'][index]
        item['faces'] = self.shapes['faces'][index]
        item['poses'] = self.shapes['poses'][index]
        item['betas'] = self.shapes['betas'][index]
        
        # get eigenfunctions/eigenvalues
        item = get_spectral_ops(item, num_evecs=self.num_evecs,)
        
        # 1 to 1 correspondence
        item['corr'] = torch.tensor(list(range(len(item['verts']))))        
        
        payload =  {
            'first': self.template,
            'second': item,
        }
        payload['second']['C_gt_xy'], payload['second']['C_gt_yx'] = \
            self.get_functional_map(payload['first'], payload['second'])
        
        return payload


    def __len__(self):
        return len(self.shapes['verts'])



        
            

            
        
        