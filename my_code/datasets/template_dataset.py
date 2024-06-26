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

# from shape_matching.my_code.datasets.surreal_legacy.surreal_dataset import get_spectral_ops

import datasets_code.shape_dataset as shape_dataset
import my_code.datasets.preprocessing as preprocessing


def get_template(template_path, num_evecs, template_corr, centering):
    
    # load template mesh
    template_mesh = trimesh.load(template_path, process=False)

    assert template_mesh is not None, f'No template_mesh found'
    
    # make the template object
    template = {
        'id': torch.tensor(-1),
        'verts': torch.tensor(template_mesh.vertices).float(),
        'faces': torch.tensor(template_mesh.faces).long(),
        'corr': torch.tensor(template_corr),
    }
    
    # center the template
    if centering == 'bbox':
        template['verts'] = preprocessing.center_bbox(template['verts'])
    elif centering == 'mean':
        template['verts'] = preprocessing.center_mean(template['verts'])
    else:
        raise RuntimeError(f'centering={centering} not recognized')
    
    # normalize vertices
    template['verts'] = preprocessing.normalize_face_area(template['verts'], template['faces'])
        
    # get spectral operators
    # directory of file template_path
    template = preprocessing.get_spectral_ops(
        template,
        num_evecs=num_evecs,
        cache_dir=os.path.dirname(template_path)
        )
    
    return template
        
    


class TemplateDataset(Dataset):
    def __init__(self,
                 base_dataset,
                 template_path,
                 template_corr,
                 num_evecs,
                 return_Cxy=True,
                 preload_base_dataset=True,
                #  canonicalize_fmap='max'
                canonicalize_fmap=None
                 ):

        self.num_evecs = num_evecs
        self.return_Cxy = return_Cxy
        self.canonicalize_fmap = canonicalize_fmap

        # cache the base dataset
        if preload_base_dataset:
            self.base_dataset = []
            for i in tqdm(range(len(base_dataset)), desc='Loading base dataset'):
                self.base_dataset.append(base_dataset[i])
                
            # self.base_dataset = base_dataset
            # for i in tqdm(range(len(self.base_dataset)), desc='Loading base dataset'):
            #     self.base_dataset[i]
                
        else:
            self.base_dataset = base_dataset
            
        assert len(self.base_dataset) > 0, f'No base_dataset found'
        
        # get the template
        self.template = get_template(
            template_path=template_path,
            num_evecs=num_evecs,
            template_corr=template_corr,
            centering=base_dataset.centering
            )
        
        
    def get_functional_map(self, data_x, data_y):

        # calculate the map
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        
        C_gt_xy = torch.linalg.lstsq(
            data_y['evecs'][data_y['corr']].to(device),
            data_x['evecs'][data_x['corr']].to(device)
            ).solution.to('cpu').unsqueeze(0)
        
        return C_gt_xy
                
        # C_gt_yx = torch.linalg.lstsq(
        #     data_x['evecs'][data_x['corr']].to(device),
        #     data_y['evecs'][data_y['corr']].to(device)
        #     ).solution.to('cpu').unsqueeze(0)

        # return C_gt_xy, C_gt_yx
        
        
    def __getitem__(self, index):
        
        # base_item = self.base_dataset[index]
        
        # item = dict()
        
        # item['id'] = torch.tensor(index)        
        # item['verts'] = base_item['verts']
        # item['faces'] = base_item['faces']
        
        # # preprocess the shape
        # item['verts'] = preprocessing.center(item['verts'])[0]
        # item['verts'] = preprocessing.scale(
        #     input_verts=item['verts'],
        #     input_faces=item['faces'],
        #     ref_verts=self.template['verts'],
        #     ref_faces=self.template['faces']
        # )[0]
        
        # item['verts_orig'] = base_item['verts']
        
        
        # # get eigenfunctions/eigenvalues
        # item = preprocessing.get_spectral_ops(item, num_evecs=self.num_evecs, cache_dir=self.cache_lb_dir)
        
        # # 1 to 1 correspondence
        # item['corr'] = base_item['corr']
        
        
        item = self.base_dataset[index]
        item['id'] = torch.tensor(index)
        
        
        payload =  {
            'first': self.template,
            'second': item,
        }
        
        if self.return_Cxy:
            # payload['second']['C_gt_xy'], payload['second']['C_gt_yx'] = \
            #     self.get_functional_map(payload['first'], payload['second'])
            
            payload['second']['C_gt_xy'] =\
                self.get_functional_map(payload['first'], payload['second'])
                
            if self.canonicalize_fmap is not None:
                
                C_gt_xy_prepr, evecs_second_prepr = preprocessing.canonicalize_fmap(
                    canon_type=self.canonicalize_fmap,
                    data_payload=payload
                    )
                
                ### doesn't work: dict from single dataset is modified during canonicalization
                # save the uncanonicalized values
                # payload['second']['C_gt_xy_uncan'] = payload['second']['C_gt_xy'].detach().clone()
                # payload['second']['evecs_uncan'] = payload['second']['evecs'].detach().clone()
                
                # assign the canonicalized values
                payload['second']['C_gt_xy'] = C_gt_xy_prepr
                payload['second']['evecs'] = evecs_second_prepr
                
                
            #     payload = preprocessing.canonicalize_fmap(
            #         canon_type=self.canonicalize_fmap,
            #         data_payload=payload
            #         )
                    
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