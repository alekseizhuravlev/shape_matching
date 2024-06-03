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

from datasets.surreal_dataset import get_spectral_ops



class TemplateDataset(Dataset):
    def __init__(self,
                 base_dataset,
                 num_evecs=200
                 ):

        self.data_root = '/home/s94zalek/shape_matching/data/SURREAL_full'
        self.num_evecs = num_evecs

        # generate male-female, straight-bent shapes
        self.base_dataset = base_dataset
        
        # load template mesh
        self.template_mesh = trimesh.load('/home/s94zalek/shape_matching/data/SURREAL_full/template/template.ply')

        # sanity check
        assert len(self.base_dataset) > 0, f'No base_dataset found'
        assert self.template_mesh is not None, f'No template_mesh found'
        
        # make the template object
        self.template = {
            'id': torch.tensor(-1),
            'verts': torch.tensor(self.template_mesh.vertices).float(),
            'faces': torch.tensor(self.template_mesh.faces).long(),
            'corr': torch.tensor(list(range(len(self.template_mesh.vertices)))),
        }
        self.template = get_spectral_ops(self.template, num_evecs=self.num_evecs)
     