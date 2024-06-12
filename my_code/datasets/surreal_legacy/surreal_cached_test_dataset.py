import os
import numpy as np
import torch


def preprocess_cached_shape_data(key, cached_data):
    
    cached_data = torch.tensor(cached_data)
        
    # if key in ['C_gt_xy', 'C_gt_yx', 'evals']:
    #     cached_data = cached_data.unsqueeze(1)
    
    return cached_data


class SurrealTestDataset(torch.utils.data.Dataset):
    def __init__(self, base_folder):
        super(SurrealTestDataset, self).__init__()
        
        self.first_shape = dict()
        # iterate over all files in base_folder/first
        for file in os.listdir(base_folder + '/first'):
            if file.endswith('.npy'):
                key = file.split('.')[0]
                
                # load the file and preprocess it
                self.first_shape[key] = np.load(base_folder + '/first/' + file)
                self.first_shape[key] = preprocess_cached_shape_data(key, self.first_shape[key])
                
                
        self.second_shapes = dict()
        # iterate over all files in base_folder/second
        for file in os.listdir(base_folder + '/second'):
            if file.endswith('.npy'):
                key = file.split('.')[0]
                
                # load the file and preprocess it
                self.second_shapes[key] = np.load(base_folder + '/second/' + file)
                self.second_shapes[key] = preprocess_cached_shape_data(key, self.second_shapes[key])

                
    def __len__(self):
        return len(self.second_shapes['evals'])
    
    def __getitem__(self, index):
        
        # iterate over all keys in first_shape and second_shapes
        # here we need to discard the batch dimension
        first_payload = dict()
        for key in self.first_shape.keys():
            first_payload[key] = self.first_shape[key][0]
        
        # get the index-th element from the second_shapes
        second_payload = dict()
        for key in self.second_shapes.keys():
            second_payload[key] = self.second_shapes[key][index]
        
        return {
            'first': first_payload,
            'second': second_payload
        }
        
        
if __name__ == '__main__':
    base_folder = '/home/s94zalek/shape_matching/data/SURREAL_full/full_datasets/dataset_158_158_316_0_32_93/test'
    
    dataset = SurrealTestDataset(base_folder)

    print('---- First shape')
    # print(dataset[2]['first']['name'])
    print(len(dataset))
    for key in dataset[2]['first'].keys():
        print(key, dataset[2]['first'][key].shape)

        
    print('---- Second')
    # print(dataset[2]['second']['name'])
    for key in dataset[2]['second'].keys():
        print(key, dataset[2]['second'][key].shape)

    