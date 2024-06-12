import torch
import numpy as np

import os
import sys
curr_dir = os.getcwd()
if 's94zalek_hpc' in curr_dir:
    user_name = 's94zalek_hpc'
else:
    user_name = 's94zalek'
sys.path.append(f'/home/{user_name}/shape_matching')

from my_code.datasets.surreal_cached_train_dataset import SurrealTrainDataset

import my_code.datasets.shape_dataset as shape_dataset
import my_code.datasets.template_dataset as template_dataset



    

def get_datasets(config):
    
    # val dataset (SURREAL)
    # val_dataset = TemplateSurrealDataset3DC(
    #     shape_path=f'/home/{user_name}/3D-CODED/data/datas_surreal_test.pth',
    #     num_evecs=config["model_params"]["sample_size"],
    #     use_cuda=False,
    #     cache_lb_dir=f'{dataset_base_folder}/{config["dataset_name"]}/test'
    # )  
    # # select 100 random samples as a subset
    # # val_dataset = torch.utils.data.Subset(val_dataset, np.random.choice(len(val_dataset), 100, replace=False))
    
    
    # # test dataset (FAUST)    
    # dataset_faust_single = shape_dataset.SingleFaustDataset(
    #     data_root='data/FAUST_original',
    #     phase='train',
    #     return_faces=True,
    #     return_evecs=False, num_evecs=32,
    #     return_corr=False, return_dist=False,
    # )
    # test_dataset = template_dataset.TemplateDataset(
    #     base_dataset=dataset_faust_single,
    #     num_evecs=32,
    #     cache_lb_dir='/home/s94zalek_hpc/shape_matching/data/FAUST_scaled/original_32'
    # ) 
    
    
    ### train dataset (SURREAL)
    
    dataset_base_folder = f'data/SURREAL_full/full_datasets'
    dataset_train = SurrealTrainDataset(f'{dataset_base_folder}/{config["dataset_name"]}/train')
    
    ### single shape datasets
    
    dataset_surreal = shape_dataset.SingleShapeDataset(
        data_root = 'data_with_smpl_corr/SURREAL_test',
        centering = 'bbox',
        num_evecs=config["model_params"]["sample_size"],
        lb_cache_dir=f'data_with_smpl_corr/SURREAL_test/{config["model_params"]["sample_size"]}'
    )   
    dataset_faust_orig = shape_dataset.SingleShapeDataset(
        data_root = 'data_with_smpl_corr/FAUST_original',
        centering = 'bbox',
        num_evecs=config["model_params"]["sample_size"],
        lb_cache_dir=f'data_with_smpl_corr/FAUST_original/{config["model_params"]["sample_size"]}'
    )
    dataset_faust_r = shape_dataset.SingleShapeDataset(
        data_root = 'data_with_smpl_corr/FAUST_r',
        centering = 'bbox',
        num_evecs=config["model_params"]["sample_size"],
        lb_cache_dir=f'data_with_smpl_corr/FAUST_r/{config["model_params"]["sample_size"]}'
    )
    dataset_faust_a = shape_dataset.SingleShapeDataset(
        data_root = 'data_with_smpl_corr/FAUST_a',
        centering = 'bbox',
        num_evecs=config["model_params"]["sample_size"],
        lb_cache_dir=f'data_with_smpl_corr/FAUST_a/{config["model_params"]["sample_size"]}'
    )
    dataset_shrec = shape_dataset.SingleShapeDataset(
        data_root = 'data_with_smpl_corr/SHREC19_original',
        centering = 'bbox',
        num_evecs=config["model_params"]["sample_size"],
        lb_cache_dir=f'data_with_smpl_corr/SHREC19_original/{config["model_params"]["sample_size"]}'
    )
    
    ### template datasets
    
    dataset_faust_orig_template = template_dataset.TemplateDataset(
        base_dataset=dataset_faust_orig,
        template_path='data/SURREAL_full/template/template.ply',
        template_corr=list(range(6890)),
        num_evecs=dataset_faust_orig.num_evecs,
    )
    dataset_faust_r_template = template_dataset.TemplateDataset(
        base_dataset=dataset_faust_r,
        template_path='data/SURREAL_full/template/template.ply',
        template_corr=np.loadtxt('data_with_smpl_corr/FAUST_r/sampleID.vts', dtype=int) - 1,
        num_evecs=dataset_faust_r.num_evecs,
    )
    dataset_faust_a_template = template_dataset.TemplateDataset(
        base_dataset=dataset_faust_a,
        template_path='data/SURREAL_full/template/template.ply',
        template_corr=list(range(6890)),
        num_evecs=dataset_faust_a.num_evecs,
    )
    dataset_shrec_template = template_dataset.TemplateDataset(
        base_dataset=dataset_shrec,
        template_path='data/SURREAL_full/template/template.ply',
        template_corr=list(range(6890)),
        num_evecs=dataset_shrec.num_evecs,
    )
    dataset_surreal_template = template_dataset.TemplateDataset(
        base_dataset=dataset_surreal,
        template_path='data/SURREAL_full/template/template.ply',
        template_corr=list(range(6890)),
        num_evecs=dataset_surreal.num_evecs,
    )
       
    ### dataloaders
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True)

    dataloader_faust_orig = torch.utils.data.DataLoader(dataset_faust_orig_template, batch_size=config["eval_batch_size"], shuffle=False)
    dataloader_faust_r = torch.utils.data.DataLoader(dataset_faust_r_template, batch_size=config["eval_batch_size"], shuffle=False)
    dataloader_faust_a = torch.utils.data.DataLoader(dataset_faust_a_template, batch_size=config["eval_batch_size"], shuffle=False)
    dataloader_shrec = torch.utils.data.DataLoader(dataset_shrec_template, batch_size=config["eval_batch_size"], shuffle=False)
    dataloader_surreal = torch.utils.data.DataLoader(dataset_surreal_template, batch_size=config["eval_batch_size"], shuffle=False)
    
    
    ### validation payload
    
    return {
        'train': {
            'name': 'train-SURREAL',
            'dataset': dataset_train,
            'dataloader': dataloader_train,
        },
        'val': [{
            'name': 'test-SURREAL',
            'dataset': dataset_surreal_template,
            'dataloader': dataloader_surreal,
        },{
            'name': 'test-FAUST-original',
            'dataset': dataset_faust_orig_template,
            'dataloader': dataloader_faust_orig,
        },{
            'name': 'test-FAUST-r',
            'dataset': dataset_faust_r_template,
            'dataloader': dataloader_faust_r,
        },{
            'name': 'test-FAUST-a',
            'dataset': dataset_faust_a_template,
            'dataloader': dataloader_faust_a,
        },{
            'name': 'test-SHREC19',
            'dataset': dataset_shrec_template,
            'dataloader': dataloader_shrec,
        }]
        }