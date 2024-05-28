import torch
import numpy as np

import sys
sys.path.append('/home/s94zalek/shape_matching')

from my_code.datasets.surreal_cached_train_dataset import SurrealTrainDataset
from my_code.datasets.surreal_cached_test_dataset import SurrealTestDataset

    
def create_train_test_loader(dataset_folder, batch_size):
        
    train_dataset = SurrealTrainDataset(f'{dataset_folder}/train')
    test_dataset = SurrealTestDataset(f'{dataset_folder}/test')
    
    print(f'Number of training samples: {len(train_dataset)}, number of test samples: {len(test_dataset)}')
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Fmap shape: {train_dataset[10][0].shape}, eval shape: {train_dataset[10][1].shape}')
    
    return train_dataset, train_dataloader, test_dataset, test_dataloader


if __name__ == '__main__':
    dataset_folder = '/home/s94zalek/shape_matching/data/SURREAL_full/full_datasets/dataset_16_16_32_0_32_80'
    batch_size = 32
    
    create_train_test_loader(dataset_folder, batch_size)