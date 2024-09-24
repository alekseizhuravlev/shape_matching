import numpy as np
import os
import torch
import time

def gather_files(data_dir, prefix, remove_after):
    
    # if f'{data_dir}/{prefix}.txt' exists, remove it
    # if os.path.exists(f'{data_dir}/{prefix}.txt'):
    #     print('Removing', f'{data_dir}/{prefix}.txt')
    #     os.remove(f'{data_dir}/{prefix}.txt')
    
    if os.path.exists(f'{data_dir}/{prefix}.pt'):
        raise RuntimeError(f'{data_dir}/{prefix}.pt already exists')
        # print('Removing', f'{data_dir}/{prefix}.pt')
        # os.remove(f'{data_dir}/{prefix}.pt')
    
    # get all files in dir in alphabetical order
    files = os.listdir(data_dir)
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
    
    data_pt = torch.tensor([])
    
    # make a file prefix.txt
    # with open(f'{data_dir}/{prefix}.txt', 'w') as f:
        # for file in sorted_files:
                        
            # with open(f'{data_dir}/{file["file"]}', 'r') as file_f:
                # lines = file_f.readlines()
                
                # print('Appending', file['file'], 'lines:', len(lines))
                
                # for line in lines:
                #     f.write(line)

    time_start = time.time()

    for file in sorted_files:
        # read the file as numpy array
        # data_i = np.loadtxt(f'{data_dir}/{file["file"]}')
        # data_i = torch.tensor(data_i)
        
        # read the file as torch tensor
        data_i = torch.load(f'{data_dir}/{file["file"]}')
        
        data_pt = torch.cat((data_pt, data_i), dim=0)
        
        time_end = time.time()
        print(f'{time_end - time_start:.2f}: Appending', file['file'], 'shape:', data_i.shape, 'total shape:', data_pt.shape)
        time_start = time_end
            
    # save the data to a .pt file
    torch.save(data_pt, f'{data_dir}/{prefix}.pt')
    
    # remove the files
    if remove_after:
        for file in sorted_files:
            print('Removing', f'{file["file"]}')
            os.remove(f'{data_dir}/{file["file"]}')
            
            
    

if __name__ == '__main__':
    
    dataset_name = 'SURREAL_64_template_remeshedSmoothed_augShapes'
    data_dir = f'/lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL/train/{dataset_name}/train'
    
    # data_dir = '/lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL_pair/pair_0.5_augShapes_signNet_remeshed_mass_6b_1ev_10_0.2_0.8'
    
    remove_after = True
    
    gather_files(data_dir, 'evals_first', remove_after)
    gather_files(data_dir, 'evals_second', remove_after)
    
    gather_files(data_dir, 'C_gt_xy', remove_after)
    gather_files(data_dir, 'C_gt_yx', remove_after)

    gather_files(data_dir, 'evecs_cond_first', remove_after)
    gather_files(data_dir, 'evecs_cond_second', remove_after)
    
    