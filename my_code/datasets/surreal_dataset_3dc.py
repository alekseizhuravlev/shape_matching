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
import os
curr_dir = os.getcwd()
if 's94zalek_hpc' in curr_dir:
    user_name = 's94zalek_hpc'
else:
    user_name = 's94zalek'
sys.path.append(f'/home/{user_name}/shape_matching')

import my_code.datasets.preprocessing as preprocessing
# from my_code.datasets.surreal_legacy.surreal_dataset import get_spectral_ops


class TemplateSurrealDataset3DC(Dataset):
    def __init__(self,
                 shape_path,
                 num_evecs,
                 use_cuda,
                 cache_lb_dir,
                 return_evecs,
                 mmap,
                 ):
        
        # raise RuntimeError("Use regular TemplateDataset")

        self.data_root = f'/home/{user_name}/shape_matching/data/SURREAL_full'
        self.num_evecs = num_evecs
        self.use_cuda = use_cuda
        self.cache_lb_dir = cache_lb_dir
        self.return_evecs = return_evecs

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
        # center the template
        # self.template['verts'] = preprocessing.center(self.template['verts'])[0]
        
        # center the template
        self.template['verts'] = preprocessing.center_bbox(self.template['verts'])
        self.template['verts'] = preprocessing.normalize_face_area(self.template['verts'], self.template['faces'])
            
        
        self.template = preprocessing.get_spectral_ops(self.template, num_evecs=self.num_evecs)
     
    
    def get_functional_map(self, data_x, data_y):

        # calculate the map
        device = 'cuda' if self.use_cuda and torch.cuda.is_available() else 'cpu'
        
        C_gt_xy = torch.linalg.lstsq(
            data_y['evecs'][data_y['corr']].to(device),
            data_x['evecs'][data_x['corr']].to(device)
            ).solution.to('cpu').unsqueeze(0)
        
        C_gt_yx = torch.linalg.lstsq(
            data_x['evecs'][data_x['corr']].to(device),
            data_y['evecs'][data_y['corr']].to(device)
            ).solution.to('cpu').unsqueeze(0)

        return C_gt_xy, C_gt_yx
        
        
    def __getitem__(self, index):
        
        item = dict()
        
        # print(index, type(index))
        item['id'] = torch.tensor(index)        
        item['verts'] = self.shapes[index]
        item['faces'] = self.template['faces']
        
        # preprocess the shape
        # item['verts'] = preprocessing.center(item['verts'])[0]
        # item['verts'] = preprocessing.scale(
        #     input_verts=item['verts'],
        #     input_faces=item['faces'],
        #     ref_verts=self.template['verts'],
        #     ref_faces=self.template['faces']
        # )[0]
        
        item['verts'] = preprocessing.center_bbox(item['verts'])
        item['verts'] = preprocessing.normalize_face_area(item['verts'], item['faces'])
        
        
        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            item = preprocessing.get_spectral_ops(item, num_evecs=self.num_evecs, cache_dir=self.cache_lb_dir)
        
        # 1 to 1 correspondence
        item['corr'] = torch.tensor(list(range(len(item['verts']))))        
        
        payload =  {
            'first': self.template,
            'second': item,
        }
        
        if self.return_evecs:
            payload['second']['C_gt_xy'], payload['second']['C_gt_yx'] = \
                self.get_functional_map(payload['first'], payload['second'])
        
        return payload


    def __len__(self):
        return len(self.shapes)



        
            

            
        
        