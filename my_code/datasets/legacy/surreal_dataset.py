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


class TemplateSurrealDataset(Dataset):
    def __init__(self,
                 phase,
                 num_evecs=200
                 ):

        self.data_root = '/home/s94zalek/shape_matching/data/SURREAL'
        self.num_evecs = num_evecs
        self.phase = phase

        # load the shapes
        if phase == 'train':
            self.shapes = torch.tensor(
                np.load(self.data_root + '/12k_shapes_train.npy', allow_pickle=True)
                ).float()
        else:
            self.shapes = torch.tensor(
                np.load(self.data_root + '/12k_shapes_test.npy', allow_pickle=True)
                ).float()
        
        # load template mesh
        self.template_mesh = trimesh.load(self.data_root + '/12ktemplate.ply')

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
        item['verts'] = self.shapes[index]
        item['faces'] = self.template['faces']
        
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
        return len(self.shapes)


if __name__ == '__main__':
    dataset = TemplateSurrealDataset(phase='train', num_evecs=50)
    
    fmap_path = '/home/s94zalek/shape_matching/data/SURREAL/functional_maps/train'
    evals_path = '/home/s94zalek/shape_matching/data/SURREAL/eigenvalues/train'
    evecs_path = '/home/s94zalek/shape_matching/data/SURREAL/eigenvectors/train'
    
    shutil.rmtree(fmap_path, ignore_errors=True)
    shutil.rmtree(evals_path, ignore_errors=True)
    shutil.rmtree(evecs_path, ignore_errors=True)
    
    os.makedirs(fmap_path, exist_ok=True)
    os.makedirs(evals_path, exist_ok=True)
    os.makedirs(evecs_path, exist_ok=True)

    lb_file =os.path.join(evals_path, 'eigenvalues.txt')
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        
        with open(os.path.join(fmap_path, f'{int(data["second"]["name"]):05d}.txt'), 'wb') as f:
            np.savetxt(f, data['C_gt_xy'].numpy().astype(np.float32))
            
        # with open(os.path.join(evecs_path, f'{int(data["second"]["name"]):05d}.txt'), 'wb') as f:
        #     np.savetxt(f, data['second']['evecs'].numpy())
            
        with open(lb_file, 'ab') as f:
            np.savetxt(f, data['second']['evals'].numpy().astype(np.float32), newline=" ")
            f.write(b'\n')
        
        