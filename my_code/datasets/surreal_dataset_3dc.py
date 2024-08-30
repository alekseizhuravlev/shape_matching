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


class TemplateSurrealDataset3DC(Dataset):
    def __init__(self,
                 shape_path,
                 num_evecs,
                 cache_lb_dir,
                 return_evecs,
                 mmap,
                 augmentations,
                 ):
        
        # raise RuntimeError("Use regular TemplateDataset")

        self.data_root = f'/home/{user_name}/shape_matching/data/SURREAL_full'
        self.num_evecs = num_evecs
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
        # center the template
        # self.template['verts'] = preprocessing.center(self.template['verts'])[0]
        
        # center the template
        self.template['verts'] = preprocessing.center_bbox(self.template['verts'])
        self.template['verts'] = preprocessing.normalize_face_area(self.template['verts'], self.template['faces'])
            
        
        self.template = preprocessing.get_spectral_ops(self.template, num_evecs=self.num_evecs)
     
    
    def get_functional_map(self, data_x, data_y):

        # calculate the map
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        C_gt_xy = torch.linalg.lstsq(
            data_y['evecs'][data_y['corr']].to(device),
            data_x['evecs'][data_x['corr']].to(device)
            ).solution.to('cpu') #.unsqueeze(0)
        
        C_gt_yx = torch.linalg.lstsq(
            data_x['evecs'][data_x['corr']].to(device),
            data_y['evecs'][data_y['corr']].to(device)
            ).solution.to('cpu') #.unsqueeze(0)

        return C_gt_xy, C_gt_yx
        
        
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
            
            # # sample the simplification percent
            # simplify_strength = np.random.uniform(
            #     self.augmentations['remesh']['simplify_strength_min'],
            #     self.augmentations['remesh']['simplify_strength_max'],
            #     )
            # # remesh and simplify the shape
            # item['verts'], item['faces'] = remesh.remesh_simplify_iso(
            #     verts_orig,
            #     faces_orig,
            #     n_remesh_iters=self.augmentations['remesh']['n_remesh_iters'],
            #     remesh_targetlen=self.augmentations['remesh']['remesh_targetlen'],
            #     simplify_strength=simplify_strength,
            # )
            
            # randomly choose the remeshing type
            remesh_type = np.random.choice(['isotropic', 'anisotropic'], p=[1-self.augmentations["remesh"]["anisotropic"]["probability"], self.augmentations["remesh"]["anisotropic"]["probability"]])
            
            if remesh_type == 'isotropic':
                simplify_strength = np.random.uniform(self.augmentations["remesh"]["isotropic"]["simplify_strength_min"], self.augmentations["remesh"]["isotropic"]["simplify_strength_max"])
                item['verts'], item['faces'] = remesh.remesh_simplify_iso(
                    verts_orig,
                    faces_orig,
                    n_remesh_iters=self.augmentations["remesh"]["isotropic"]["n_remesh_iters"],
                    remesh_targetlen=self.augmentations["remesh"]["isotropic"]["remesh_targetlen"],
                    simplify_strength=simplify_strength,
                )
            else:
                fraction_to_simplify = np.random.uniform(self.augmentations["remesh"]["anisotropic"]["fraction_to_simplify_min"], self.augmentations["remesh"]["anisotropic"]["fraction_to_simplify_max"])
                simplify_strength = np.random.uniform(self.augmentations["remesh"]["anisotropic"]["simplify_strength_min"], self.augmentations["remesh"]["anisotropic"]["simplify_strength_max"])
                
                item['verts'], item['faces'] = remesh.remesh_simplify_anis(
                    verts_orig,
                    faces_orig,
                    n_remesh_iters=self.augmentations["remesh"]["anisotropic"]["n_remesh_iters"],
                    fraction_to_simplify=fraction_to_simplify,
                    simplify_strength=simplify_strength,
                    weighted_by=self.augmentations["remesh"]["anisotropic"]["weighted_by"]
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



        
            

            
        
        