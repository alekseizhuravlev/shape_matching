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

from my_code.datasets.surreal_dataset import get_spectral_ops

import datasets_code.shape_dataset as shape_dataset
import my_code.datasets.preprocessing as preprocessing



class TemplateDataset(Dataset):
    def __init__(self,
                 base_dataset,
                 num_evecs,
                 cache_lb_dir,
                 return_Cxy
                 ):

        self.data_root = f'/home/{user_name}/shape_matching/data/SURREAL_full'
        self.num_evecs = num_evecs
        self.cache_lb_dir = cache_lb_dir
        self.return_Cxy = return_Cxy

        # cache the base dataset
        self.base_dataset = []
        for i in tqdm(range(len(base_dataset)), desc='Loading base dataset'):
            self.base_dataset.append(base_dataset[i])
        
        # load template mesh
        self.template_mesh = trimesh.load(f'/home/{user_name}/shape_matching/data/SURREAL_full/template/template.ply', process=False)

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
        # center the template
        self.template['verts'] = preprocessing.center(self.template['verts'])[0]
        
        self.template = get_spectral_ops(self.template, num_evecs=self.num_evecs)
        
        
    def get_functional_map(self, data_x, data_y):

        # calculate the map
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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
        
        base_item = self.base_dataset[index]
        
        item = dict()
        
        item['id'] = torch.tensor(index)        
        item['verts'] = base_item['verts']
        item['faces'] = base_item['faces']
        
        # preprocess the shape
        item['verts'] = preprocessing.center(item['verts'])[0]
        item['verts'] = preprocessing.scale(
            input_verts=item['verts'],
            input_faces=item['faces'],
            ref_verts=self.template['verts'],
            ref_faces=self.template['faces']
        )[0]
        
        # get eigenfunctions/eigenvalues
        item = get_spectral_ops(item, num_evecs=self.num_evecs, cache_dir=self.cache_lb_dir)
        
        # 1 to 1 correspondence
        item['corr'] = base_item['corr']
        
        
        payload =  {
            'first': self.template,
            'second': item,
        }
        
        if self.return_Cxy:
            payload['second']['C_gt_xy'], payload['second']['C_gt_yx'] = \
                self.get_functional_map(payload['first'], payload['second'])
                    
        return payload


    def __len__(self):
        return len(self.base_dataset)
    
    
    
    
if __name__ == '__main__':
    
    dataset_faust_single = shape_dataset.SingleFaustDataset(
        data_root='data/FAUST_original',
        phase='train',
        return_faces=True,
        return_evecs=True, num_evecs=32,
        return_corr=True, return_dist=False
    )
    
    dataset_template = TemplateDataset(
        base_dataset=dataset_faust_single,
        num_evecs=32
    )
    
    print(dataset_template[10])