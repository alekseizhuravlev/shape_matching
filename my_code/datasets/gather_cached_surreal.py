import numpy as np
import os


def gather_files(data_dir, prefix):
    
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
    
    # make a file prefix.txt
    with open(f'{data_dir}/{prefix}.txt', 'w') as f:
        for file in sorted_files:
                        
            with open(f'{data_dir}/{file["file"]}', 'r') as file_f:
                lines = file_f.readlines()
                
                print('Appending', file['file'], 'lines:', len(lines))
                
                for line in lines:
                    f.write(line)
            
    

if __name__ == '__main__':
    data_dir = '/home/s94zalek_hpc/shape_matching/data/SURREAL_full/full_datasets/dataset_SURREAL_train_withAug_productSuppCond_32_4block_50k_pth/train'
    gather_files(data_dir, 'evals')
    
    gather_files(data_dir, 'C_gt_xy')

    gather_files(data_dir, 'evecs_cond_first')
    gather_files(data_dir, 'evecs_cond_second')
    
    