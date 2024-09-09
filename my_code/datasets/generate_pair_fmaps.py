import shutil
from tqdm import tqdm
import torch
import os
import time

def process_evecs(evecs_second_with_corr, evecs_cond, n_pairs):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    C_xy_evecs_list = []   
    evecs_cond_first_list = [] 
    evecs_cond_second_list = []

    for first_idx in tqdm(range(evecs_second_with_corr.shape[0])):
        
        # select the first evecs
        evecs_first = evecs_second_with_corr[first_idx]
        
        for curr_pair in range(n_pairs):
            
            # sample a random second_idx
            second_idx = torch.randint(evecs_second_with_corr.shape[0], (1,)).item()
            while second_idx == first_idx:
                second_idx = torch.randint(evecs_second_with_corr.shape[0], (1,)).item()
                
            # select the second evecs
            evecs_second = evecs_second_with_corr[second_idx]
            
            # compute the fmap
            C_xy_evecs = torch.linalg.lstsq(
                evecs_second.to(device),
                evecs_first.to(device)
                ).solution.cpu()
            
            C_xy_evecs_list.append(C_xy_evecs)
            evecs_cond_first_list.append(evecs_cond[first_idx])
            evecs_cond_second_list.append(evecs_cond[second_idx])
            
    C_xy_evecs_tensor = torch.stack(C_xy_evecs_list, dim=0)
    evecs_cond_first_tensor = torch.stack(evecs_cond_first_list, dim=0)
    evecs_cond_second_tensor = torch.stack(evecs_cond_second_list, dim=0)
    
    return C_xy_evecs_tensor, evecs_cond_first_tensor, evecs_cond_second_tensor


def get_files_with_prefix(data_dir_in, prefix):
    
    # get all files in dir in alphabetical order
    files = os.listdir(data_dir_in)
    files = sorted(files)
    
    files_with_start_end = []
    for file in files:
        if prefix in file:
            file_no_prefix = file.replace(f'{prefix}_', '')
            # remove extension
            file_no_prefix = file_no_prefix.split('.')[0]
            
            # get the start and end indices
            start_idx, end_idx = file_no_prefix.split('_')
            
            files_with_start_end.append({
                    'file': file,
                    'start_idx': int(start_idx),
                    'end_idx': int(end_idx)
            })
            
    sorted_files = sorted(files_with_start_end, key=lambda x: x['start_idx'])
    
    return sorted_files
    

def full_pipeline(data_dir_in, data_dir_out, dataset_name, n_pairs):

    if os.path.exists(f'{data_dir_out}/{dataset_name}'):
        # raise RuntimeError(f'{data_dir_out}/{dataset_name} already exists')
        
        # make a prompt to remove the directory
        print(f'{data_dir_out}/{dataset_name} already exists')
        print('Press y to remove, any other key to exit')
        
        user_input = input()
        
        if user_input == 'y':
            print('Removing', f'{data_dir_out}/{dataset_name}')
            shutil.rmtree(f'{data_dir_out}/{dataset_name}')
    
    os.makedirs(f'{data_dir_out}/{dataset_name}')
    
    files_evecs_second_with_corr = get_files_with_prefix(
        data_dir_in, 'evecs_second_with_corr')
    files_evecs_cond = get_files_with_prefix(
        data_dir_in, 'evecs_cond_second')
    
    
    # time_start = time.time()
    
    C_xy_evecs_full = []
    evecs_cond_first_full = []
    evecs_cond_second_full = []

    for i in range(len(files_evecs_second_with_corr)):
        
        print(f'{i}) Processing indices',
              files_evecs_second_with_corr[i]['start_idx'], files_evecs_second_with_corr[i]['end_idx'])
        
        assert files_evecs_second_with_corr[i]['start_idx'] == files_evecs_cond[i]['start_idx']
        assert files_evecs_second_with_corr[i]['end_idx'] == files_evecs_cond[i]['end_idx']
        
        evecs_second_with_corr_i = torch.load(f'{data_dir_in}/{files_evecs_second_with_corr[i]["file"]}')
        evecs_cond_i = torch.load(f'{data_dir_in}/{files_evecs_cond[i]["file"]}')
        
        C_xy_evecs_i, evecs_cond_first_i, evecs_cond_second_i = process_evecs(
            evecs_second_with_corr_i, evecs_cond_i, n_pairs)
        
        # print the size of tensors in GB
        print(f'C_xy_evecs_i: {C_xy_evecs_i.shape}, {C_xy_evecs_i.element_size() * C_xy_evecs_i.nelement() / 1e9} GB')
        print(f'evecs_cond_first_i: {evecs_cond_first_i.shape}, {evecs_cond_first_i.element_size() * evecs_cond_first_i.nelement() / 1e9} GB')
        print(f'evecs_cond_second_i: {evecs_cond_second_i.shape}, {evecs_cond_second_i.element_size() * evecs_cond_second_i.nelement() / 1e9} GB')
        
        C_xy_evecs_full.append(C_xy_evecs_i)
        evecs_cond_first_full.append(evecs_cond_first_i)
        evecs_cond_second_full.append(evecs_cond_second_i)
                
    C_xy_evecs_full = torch.stack(C_xy_evecs_full, dim=0)
    evecs_cond_first_full = torch.stack(evecs_cond_first_full, dim=0)
    evecs_cond_second_full = torch.stack(evecs_cond_second_full, dim=0)
    
    # save the data to a .pt file
    torch.save(C_xy_evecs_full, f'{data_dir_out}/{dataset_name}/C_gt_xy.pt')
    torch.save(evecs_cond_first_full, f'{data_dir_out}/{dataset_name}/evecs_cond_first.pt')
    torch.save(evecs_cond_second_full, f'{data_dir_out}/{dataset_name}/evecs_cond_second.pt')
    
    
if __name__ == '__main__':
        
    data_dir_in = '/lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL/train/SURREAL_evecs_10_augShapes_signNet_remeshed_mass_6b_1ev_10_0.2_0.8/train'
    data_dir_out = '/lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL_pair'
    
    dataset_name = 'pair_10_augShapes_signNet_remeshed_mass_6b_1ev_10_0.2_0.8'
    
    
    full_pipeline(data_dir_in, data_dir_out, dataset_name, 10)
    