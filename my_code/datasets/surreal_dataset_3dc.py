import os
import numpy as np
import trimesh

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import sys
import os
# curr_dir = os.getcwd()
# if 's94zalek_hpc' in curr_dir:
#     user_name = 's94zalek_hpc'
# else:
#     user_name = 's94zalek'
# sys.path.append(f'/home/{user_name}/shape_matching')

import my_code.datasets.preprocessing as preprocessing
# from my_code.datasets.surreal_legacy.surreal_dataset import get_spectral_ops
import my_code.sign_canonicalization.remesh as remesh
import utils.fmap_util as fmap_util

import copy


class TemplateSurrealDataset3DC(Dataset):
    def __init__(self,
                 shape_path,
                 num_evecs,
                 cache_lb_dir,
                 return_evecs,
                 return_fmap,
                 mmap,
                 augmentations,
                 template_path,
                 template_corr,
                 mesh_orig_faces_path,
                 centering,
                 return_shot=False,
                 ):
        
        self.num_evecs = num_evecs
        self.cache_lb_dir = cache_lb_dir
        self.return_evecs = return_evecs
        self.return_fmap = return_fmap
        self.augmentations = augmentations
        self.centering = centering
        self.return_shot = return_shot
        
        # determine if augmentations are partial
        if self.augmentations is not None and 'remesh' in self.augmentations:
            if 'partial' in self.augmentations['remesh']:
                self.partial = True
            else:
                self.partial = False
        else:
            self.partial = False

        # load the shapes from 3D-coded
        self.shapes = torch.load(shape_path, mmap=mmap)
        
        # load template mesh
        self.template_mesh = trimesh.load(
            template_path
        )
        self.mesh_orig_faces = trimesh.load(
            mesh_orig_faces_path
        )
        self.faces = torch.tensor(self.mesh_orig_faces.faces).int()
        # f'/home/{user_name}/shape_matching/data/SURREAL_full/template/template.ply'

        # sanity check
        assert len(self.shapes) > 0, f'No shapes found'
        assert self.template_mesh is not None, f'No template_mesh found'
        
        # make the template object
        self.template = {
            'id': torch.tensor(-1),
            'verts': torch.tensor(self.template_mesh.vertices).float(),
            'faces': torch.tensor(self.template_mesh.faces).int(),
            'corr': torch.tensor(template_corr),
            # 'corr': torch.tensor(list(range(len(self.template_mesh.vertices)))),
        }
        # center the template
        # self.template['verts'] = preprocessing.center(self.template['verts'])[0]
        
        # center the template
        
        if self.centering == 'bbox':
            self.template['verts'] = preprocessing.center_bbox(self.template['verts'])
        elif self.centering == 'mean':
            self.template['verts'] = preprocessing.center_mean(self.template['verts'])
        else:
            raise ValueError(f'Invalid centering method: {self.centering}')
        
        
        self.template['verts'] = preprocessing.normalize_face_area(self.template['verts'], self.template['faces'])
            
        
        self.template = preprocessing.get_spectral_ops(self.template, num_evecs=self.num_evecs)
        
        if self.return_shot:
            import pyshot
            
            self.template['shot'] = torch.tensor(
                pyshot.get_descriptors(
                    self.template['verts'].numpy().astype(np.double),
                    self.template['faces'].numpy().astype(np.int64),
                    radius=100,
                    local_rf_radius=100,
                    
                    min_neighbors=3,
                    n_bins=10,
                    double_volumes_sectors=True,
                    use_interpolation=True,
                    use_normalization=True,
                ), dtype=torch.float32)
            
     
    
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
        
        n_attempts = 10
        
        while n_attempts > 0:
            try:
                return self._get_item(index)
            except Exception as e:
                print(f'Error: {e}')
                n_attempts -= 1
                print(f'Attempts left: {n_attempts}')
                
        raise ValueError(f'Failed to get item {index}')
        
        
    def _get_item(self, index):
        
        item = dict()
        
        item['id'] = torch.tensor(index)        
        item['verts'] = self.shapes[index]
        item['faces'] = self.faces
        
        # augmentations
        if self.augmentations is not None and 'remesh' in self.augmentations:
            verts_orig = item['verts']
            faces_orig = item['faces']
                        
            if self.partial:
                item['verts'], item['faces'], item['corr'] = remesh.augmentation_pipeline_partial(
                    verts_orig,
                    faces_orig,
                    self.augmentations,
                )
            else:   
                item['verts'], item['faces'], item['corr'] = remesh.augmentation_pipeline(
                    verts_orig,
                    faces_orig,
                    self.augmentations,
                )       
        else:
            # 1 to 1 correspondence
            item['corr'] = torch.tensor(list(range(len(item['verts']))))        
        
        # center the shape and normalize the face area
        
        if self.centering == 'bbox':
            item['verts'] = preprocessing.center_bbox(item['verts'])
        elif self.centering == 'mean':
            item['verts'] = preprocessing.center_mean(item['verts'])
        else:
            raise ValueError(f'Invalid centering method: {self.centering}')
        
        item['verts'] = preprocessing.normalize_face_area(item['verts'], item['faces'])
        
        
        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            item = preprocessing.get_spectral_ops(item, num_evecs=self.num_evecs, cache_dir=self.cache_lb_dir)

        
        if self.return_shot:
            
            import pyshot
            
            item['shot'] = torch.tensor(
                pyshot.get_descriptors(
                item['verts'].numpy().astype(np.double),
                item['faces'].numpy().astype(np.int64),
                radius=100,
                local_rf_radius=100,
                
                min_neighbors=3,
                n_bins=10,
                double_volumes_sectors=True,
                use_interpolation=True,
                use_normalization=True,
            ), dtype=torch.float32)
        
        
        
        if self.partial:
            payload =  {
                'first': copy.deepcopy(self.template),
                'second': item,
            }
            # now, 'first' has the correspondence: remeshed template -> smpl
            # 'second' has the correspondence: smpl -> partial shape            
            
            payload['first']['corr'] = payload['first']['corr'][payload['second']['corr']]
            payload['second']['corr'] = torch.tensor(list(range(len(payload['second']['corr']))), dtype=torch.int32)
            
            # now, 'first' has the correspondence: remeshed template -> partial shape
            # 'second' has the correspondence: partial shape -> partial shape
            # this follows the structure of SHREC16 dataset
            
        else:        
            payload =  {
                'first': copy.deepcopy(self.template),
                'second': item,
            }
            # now, 'first' has the correspondence: remeshed template -> smpl
            # 'second' has the correspondence: remeshed shape -> smpl
        
        if self.return_fmap:
            payload['second']['C_gt_xy'], payload['second']['C_gt_yx'] = \
                self.get_functional_map(payload['first'], payload['second'])
        
        return payload


    def __len__(self):
        return len(self.shapes)



        
            

            
        
        