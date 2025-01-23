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

import my_code.datasets.shape_dataset as shape_dataset
import my_code.datasets.template_dataset as template_dataset
import time

def get_val_dataset(name, phase, num_evecs, preload, return_evecs, centering, canonicalize_fmap=None,
                    recompute_evecs=False):
    
    if recompute_evecs:
        curr_time = time.time()
        lb_base_dir = f'/lustre/mlnvme/data/s94zalek_hpc-shape_matching/lb_cache/{curr_time}'
    else:
        lb_base_dir = 'data'    
    
    
    if name == 'SURREAL':
        dataset_single = shape_dataset.SingleShapeDataset(
            data_root = 'data_with_smpl_corr/SURREAL_test',
            centering = centering,
            num_evecs=num_evecs,
            lb_cache_dir=f'data_with_smpl_corr/SURREAL_test/diffusion',
            return_evecs=return_evecs
        )   
        dataset_pair = template_dataset.TemplateDataset(
            base_dataset=dataset_single,
            template_path='data/SURREAL_full/template/template.ply',
            template_corr=list(range(6890)),
            num_evecs=dataset_single.num_evecs,
            preload_base_dataset=preload,
            canonicalize_fmap=canonicalize_fmap,
            centering=centering,
            return_Cxy=return_evecs
        )
    elif name == 'FAUST_orig':
        dataset_single = shape_dataset.SingleFaustDataset(
            phase=phase,
            data_root = 'data_with_smpl_corr/FAUST_original',
            centering = centering,
            num_evecs=num_evecs,
            lb_cache_dir=f'data_with_smpl_corr/FAUST_original/diffusion',
            return_evecs=return_evecs,
        )
        dataset_pair = template_dataset.TemplateDataset(
            base_dataset=dataset_single,
            template_path='data/SURREAL_full/template/template.ply',
            template_corr=list(range(6890)),
            num_evecs=dataset_single.num_evecs,
            preload_base_dataset=preload,
            canonicalize_fmap=canonicalize_fmap,
            centering=centering,
            return_Cxy=return_evecs,
        )
    elif name == 'FAUST_r':
        dataset_single = shape_dataset.SingleFaustDataset(
            phase=phase,
            data_root = 'data_with_smpl_corr/FAUST_r',
            centering = centering,
            num_evecs=num_evecs,
            lb_cache_dir=f'data_with_smpl_corr/FAUST_r/diffusion',
            return_evecs=return_evecs,
        )
        dataset_pair = template_dataset.TemplateDataset(
            base_dataset=dataset_single,
            template_path='data/SURREAL_full/template/template.ply',
            template_corr=np.loadtxt('data_with_smpl_corr/FAUST_r/sampleID.vts', dtype=int) - 1,
            num_evecs=dataset_single.num_evecs,
            preload_base_dataset=preload,
            canonicalize_fmap=canonicalize_fmap,
            centering=centering,
            return_Cxy=return_evecs,
        )
    elif name == 'FAUST_a':
        dataset_single = shape_dataset.SingleShapeDataset(
            data_root = 'data_with_smpl_corr/FAUST_a',
            centering = centering,
            num_evecs=num_evecs,
            lb_cache_dir=f'data_with_smpl_corr/FAUST_a/diffusion',
            return_evecs=return_evecs,
        )
        dataset_pair = template_dataset.TemplateDataset(
            base_dataset=dataset_single,
            template_path='data/SURREAL_full/template/template.ply',
            template_corr=list(range(6890)),
            num_evecs=dataset_single.num_evecs,
            preload_base_dataset=preload,
            canonicalize_fmap=canonicalize_fmap,
            centering=centering,
            return_Cxy=return_evecs,
        )
    elif name == 'SHREC19_orig':
        dataset_single = shape_dataset.SingleShapeDataset(
            data_root = 'data_with_smpl_corr/SHREC19_original',
            centering = centering,
            num_evecs=num_evecs,
            lb_cache_dir=f'data_with_smpl_corr/SHREC19_original/diffusion',
            return_evecs=return_evecs,
            # lb_cache_dir=None
        )
        dataset_pair = template_dataset.TemplateDataset(
            base_dataset=dataset_single,
            template_path='data/SURREAL_full/template/template.ply',
            template_corr=list(range(6890)),
            num_evecs=dataset_single.num_evecs,
            preload_base_dataset=preload,
            canonicalize_fmap=canonicalize_fmap,
            centering=centering,
            return_Cxy=return_evecs,
        )
    elif name == 'SHREC19_r':
        dataset_single = shape_dataset.SingleShrec19Dataset(
            data_root = 'data/SHREC19_r',
            centering = centering,
            num_evecs=num_evecs,
            lb_cache_dir=f'data/SHREC19_r/diffusion',
            return_evecs=return_evecs,
            return_corr=False,
        )
        dataset_pair = None
        
    ############################################################
    # Pair datasets
    ############################################################
        
    elif name == 'FAUST_orig_pair':
        dataset_single = shape_dataset.SingleFaustDataset(
            phase=phase,
            data_root = 'data_with_smpl_corr/FAUST_original',
            centering = centering,
            num_evecs=num_evecs,
            lb_cache_dir=f'data_with_smpl_corr/FAUST_original/diffusion',
            return_evecs=return_evecs,
        )
        dataset_pair = shape_dataset.PairShapeDataset(
            dataset=dataset_single,
            cache_base_dataset=preload,
        )
        
    elif name == 'FAUST_r_pair':
        dataset_single = shape_dataset.SingleFaustDataset(
            phase=phase,
            data_root = 'data/FAUST_r',
            centering = centering,
            num_evecs=num_evecs,
            # lb_cache_dir=f'data/FAUST_r/diffusion',
            lb_cache_dir=f'{lb_base_dir}/FAUST_r/diffusion',
            return_evecs=return_evecs,
        )
        dataset_pair = shape_dataset.PairShapeDataset(
            dataset=dataset_single,
            cache_base_dataset=preload,
        )
    
    elif name == 'FAUST_a_pair':
        dataset_single = shape_dataset.SingleShapeDataset(
            data_root = 'data/FAUST_a',
            centering = centering,
            num_evecs=num_evecs,
            # lb_cache_dir=f'data/FAUST_a/diffusion',
            lb_cache_dir=f'{lb_base_dir}/FAUST_a/diffusion',
            return_evecs=return_evecs,
        )
        dataset_pair = shape_dataset.PairShapeDataset(
            dataset=dataset_single,
            cache_base_dataset=preload,
        )
        
    elif name == 'SHREC19_r_pair':
        dataset_single = shape_dataset.SingleShapeDataset(
            data_root = 'data/SHREC19_r',
            centering = centering,
            num_evecs=num_evecs,
            # lb_cache_dir=f'data/SHREC19_r/diffusion',
            lb_cache_dir=f'{lb_base_dir}/SHREC19_r/diffusion',
            return_evecs=return_evecs,
            return_corr=False,
        )
        dataset_pair = shape_dataset.PairShrec19Dataset(
            dataset=dataset_single,
            phase=phase,
        )
        
    elif name == 'SCAPE_r_pair':
        dataset_single = shape_dataset.SingleScapeDataset(
            phase=phase,
            data_root = 'data/SCAPE_r',
            centering = centering,
            num_evecs=num_evecs,
            # lb_cache_dir=f'data/SCAPE_r/diffusion',
            lb_cache_dir=f'{lb_base_dir}/SCAPE_r/diffusion',
            return_evecs=return_evecs,
        )
        dataset_pair = shape_dataset.PairShapeDataset(
            dataset=dataset_single,
            cache_base_dataset=preload,
        )
        
    elif name == 'SCAPE_a_pair':
        dataset_single = shape_dataset.SingleShapeDataset(
            data_root = 'data/SCAPE_a',
            centering = centering,
            num_evecs=num_evecs,
            # lb_cache_dir=f'data/SCAPE_a/diffusion',
            lb_cache_dir=f'{lb_base_dir}/SCAPE_a/diffusion',
            return_evecs=return_evecs,
        )
        dataset_pair = shape_dataset.PairShapeDataset(
            dataset=dataset_single,
            cache_base_dataset=preload,
        )
        
    elif name == 'DT4D_intra_pair':
        dataset_single = shape_dataset.SingleDT4DDataset(
            phase=phase,
            data_root = 'data/DT4D_r',
            centering = centering,
            num_evecs=num_evecs,
            # lb_cache_dir=f'data/DT4D_r/diffusion',
            lb_cache_dir=f'{lb_base_dir}/DT4D_r/diffusion',
            return_evecs=return_evecs,
            )
        dataset_pair = shape_dataset.PairDT4DDataset(
            dataset=dataset_single,
            inter_class=False,
            cache_base_dataset=preload,
            )
        
    elif name == 'DT4D_inter_pair':
        dataset_single = shape_dataset.SingleDT4DDataset(
            phase=phase,
            data_root='data/DT4D_r',
            centering=centering,
            num_evecs=num_evecs,
            # lb_cache_dir=f'data/DT4D_r/diffusion',
            lb_cache_dir=f'{lb_base_dir}/DT4D_r/diffusion',
            return_evecs=return_evecs,
            )
        dataset_pair = shape_dataset.PairDT4DDataset(
            dataset=dataset_single,
            inter_class=True,
            cache_base_dataset=preload,
        )
        
    elif name == 'SMAL_cat_pair':
        dataset_single = shape_dataset.SingleSmalDataset(
            phase=phase,
            category=True,
            data_root='data/SMAL_r',
            centering=centering,
            num_evecs=num_evecs,
            lb_cache_dir=f'{lb_base_dir}/SMAL_r/diffusion',
            return_evecs=return_evecs,
            )
        
        dataset_pair = shape_dataset.PairSmalDataset(
            dataset=dataset_single,
            cache_base_dataset=preload,
        )
        
    elif name == 'SMAL_nocat_pair':
        dataset_single = shape_dataset.SingleSmalDataset(
            phase=phase,
            category=False,
            data_root='data/SMAL_r',
            centering=centering,
            num_evecs=num_evecs,
            lb_cache_dir=f'{lb_base_dir}/SMAL_r/diffusion',
            return_evecs=return_evecs,
            )
        
        dataset_pair = shape_dataset.PairSmalDataset(
            dataset=dataset_single,
            cache_base_dataset=preload,
        )
        
    ############################################################
    # Partial datasets
    ############################################################
    
    elif name == 'SHREC16_cuts_pair':
        
        dataset_pair = shape_dataset.PairShrec16Dataset(
            'data/SHREC16_test/' if phase == 'test' else 'data/SHREC16/',
            categories=['david', 'michael', 'victoria'],
            cut_type='cuts', return_faces=True,
            return_evecs=return_evecs, num_evecs=num_evecs,
            return_corr=True, return_dist=False
        )
        
        data_single = [dataset_pair[i]['second'] for i in range(len(dataset_pair))]
        dataset_single = shape_dataset.DatasetFromListOfDicts(data_single)

    elif name == 'SHREC16_cuts_pair_noSingle':
        
        dataset_pair = shape_dataset.PairShrec16Dataset(
            'data/SHREC16_test/' if phase == 'test' else 'data/SHREC16/',
            categories=['david', 'michael', 'victoria'],
            cut_type='cuts', return_faces=True,
            return_evecs=return_evecs, num_evecs=num_evecs,
            return_corr=True, return_dist=False
        )
       
        dataset_single = None
        
    elif name == 'SHREC16_holes_pair':
        
        dataset_pair = shape_dataset.PairShrec16Dataset(
            'data/SHREC16_test/' if phase == 'test' else 'data/SHREC16/',
            categories=['david', 'michael', 'victoria'],
            cut_type='holes', return_faces=True,
            return_evecs=return_evecs, num_evecs=num_evecs,
            return_corr=True, return_dist=False
        )
        
        data_single = [dataset_pair[i]['second'] for i in range(len(dataset_pair))]
        dataset_single = shape_dataset.DatasetFromListOfDicts(data_single)        
    
    elif name == 'SHREC16_holes_pair_noSingle':
        
        dataset_pair = shape_dataset.PairShrec16Dataset(
            'data/SHREC16_test/' if phase == 'test' else 'data/SHREC16/',
            categories=['david', 'michael', 'victoria'],
            cut_type='holes', return_faces=True,
            return_evecs=return_evecs, num_evecs=num_evecs,
            return_corr=True, return_dist=False
        )

        dataset_single = None
        
        
    else:
        raise ValueError(f'Unknown dataset name: {name}')
        
    return dataset_single, dataset_pair
    