import os
import numpy as np
import trimesh

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import sys
import os
curr_dir = os.getcwd()
if 's94zalek_hpc' in curr_dir:
    user_name = 's94zalek_hpc'
else:
    user_name = 's94zalek'
sys.path.append(f'/home/{user_name}/shape_matching')

import my_code.datasets.preprocessing as preprocessing
# from my_code.datasets.surreal_legacy.surreal_dataset import get_spectral_ops
import my_code.sign_canonicalization.remesh as remesh
import utils.fmap_util as fmap_util


class SingleSurrealDataset(Dataset):
    def __init__(self,
                 shape_path,
                 num_evecs,
                 use_cuda,
                 cache_lb_dir,
                 return_evecs,
                 mmap,
                 augmentations,
                 ):
        
        # raise RuntimeError("Use regular TemplateDataset")

        self.data_root = f'/home/{user_name}/shape_matching/data/SURREAL_full'
        self.num_evecs = num_evecs
        self.use_cuda = use_cuda
        self.cache_lb_dir = cache_lb_dir
        self.return_evecs = return_evecs
        self.augmentations = augmentations

        # load the shapes from 3D-coded
        self.shapes = torch.load(shape_path, mmap=mmap)
        
        # load template mesh
        self.template_mesh = trimesh.load(f'/home/{user_name}/shape_matching/data/SURREAL_full/template/template.ply')

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
            
        
    def __getitem__(self, index):
        
        item = dict()
        
        # print(index, type(index))
        item['id'] = torch.tensor(index)        
        item['verts'] = self.shapes[index]
        item['faces'] = self.template['faces']
        

        # augmentations
        if self.augmentations is not None and 'remesh' in self.augmentations:
            
            verts_orig = item['verts']
            faces_orig = item['faces']
            
            # sample the simplification percent
            simplify_strength = np.random.uniform(
                self.augmentations['remesh']['simplify_strength_min'],
                self.augmentations['remesh']['simplify_strength_max'],
                )
            # remesh and simplify the shape
            item['verts'], item['faces'] = remesh.remesh_simplify_iso(
                verts_orig,
                faces_orig,
                n_remesh_iters=self.augmentations['remesh']['n_remesh_iters'],
                remesh_targetlen=self.augmentations['remesh']['remesh_targetlen'],
                simplify_strength=simplify_strength,
            )
            
            # correspondence by a nearest neighbor search
            item['corr'] = fmap_util.nn_query(
                item['verts'],
                verts_orig, 
                )
        else:
            # 1 to 1 correspondence
            item['corr'] = torch.tensor(list(range(len(item['verts']))))        
        
        # center the shape and normalize the face area
        item['verts'] = preprocessing.center_bbox(item['verts'])
        item['verts'] = preprocessing.normalize_face_area(item['verts'], item['faces'])

        
        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            item = preprocessing.get_spectral_ops(item, num_evecs=self.num_evecs, cache_dir=self.cache_lb_dir)
        
        return item


    def __len__(self):
        return len(self.shapes)


class PairSurrealDataset(Dataset):
    
    def __init__(self,
                 single_dataset,
                 ):
        self.single_dataset = single_dataset
        
        
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
        
        assert len(index) == 2, f'Expected a pair of indices, got {index}'
        
        index1, index2 = index
    
        item_1 = self.single_dataset[index1]
        item_2 = self.single_dataset[index2]
        
        payload =  {
            'first': item_1,
            'second': item_2,
        }
        
        if self.single_dataset.return_evecs:
            payload['second']['C_gt_xy'], payload['second']['C_gt_yx'] = \
                self.get_functional_map(payload['first'], payload['second'])
        
        return payload
    
    
    def __len__(self):
        return len(self.single_dataset)


        
            

            
        
        