import torch
import numpy as np


class SurrealTrainDataset(torch.utils.data.Dataset):
    def __init__(self, base_folder, fmap_direction, fmap_input_type, conditioning_types,
                 mmap):
        super(SurrealTrainDataset, self).__init__()
        
        self.fmap_input_type = fmap_input_type
        self.conditioning_types = conditioning_types
        self.mmap = mmap
        
        # load the functional maps
        if fmap_direction == 'xy':
            # self.fmaps = np.loadtxt(f'{base_folder}/C_gt_xy.txt') 
            
            self.fmaps = torch.load(f'{base_folder}/C_gt_xy.pt', mmap=self.mmap)
            
        elif fmap_direction == 'yx':
            # self.fmaps = np.loadtxt(f'{base_folder}/C_gt_yx.txt')
            
            self.fmaps = torch.load(f'{base_folder}/C_gt_yx.pt', mmap=self.mmap)
    
        assert self.fmaps.dtype == torch.float32, f'fmaps dtype: {self.fmaps.dtype}'
        
        self.fmap_dim = int(self.fmaps.shape[1])
        print('Train dataset, functional map dimension:', self.fmap_dim, 'fmaps shape:', self.fmaps.shape)
        # self.fmaps = torch.tensor(self.fmaps, dtype=torch.float32).reshape(len(self.fmaps), fmap_dim, fmap_dim)
        
        # optional: take the absolute value
        if self.fmap_input_type == 'abs':
            self.fmaps = self.fmaps.abs()
        
        if 'evals' in conditioning_types or 'evals_inv' in conditioning_types:
            # self.evals = np.loadtxt(f'{base_folder}/evals.txt')
            # self.evals = torch.tensor(self.evals, dtype=torch.float32)
            
            self.evals = torch.load(f'{base_folder}/evals.pt', mmap=self.mmap)
            
            assert self.evals.dtype == torch.float32, f'evals dtype: {self.evals.dtype}'

        if 'evecs' in conditioning_types:
            
            # print('WARNING evecs shape [32, 8] is hard coded')
            
            # self.evecs_cond_first = np.loadtxt(f'{base_folder}/evecs_cond_first.txt')
            # self.evecs_cond_first = torch.tensor(self.evecs_cond_first, dtype=torch.float32)
            # self.evecs_cond_first = self.evecs_cond_first.reshape(len(self.evecs_cond_first), fmap_dim, fmap_dim)   
            
            # self.evecs_cond_second = np.loadtxt(f'{base_folder}/evecs_cond_second.txt')
            # self.evecs_cond_second = torch.tensor(self.evecs_cond_second, dtype=torch.float32)
            # self.evecs_cond_second = self.evecs_cond_second.reshape(len(self.evecs_cond_second), fmap_dim, fmap_dim)
            
            self.evecs_cond_first = torch.load(f'{base_folder}/evecs_cond_first.pt', mmap=self.mmap)
            self.evecs_cond_second = torch.load(f'{base_folder}/evecs_cond_second.pt', mmap=self.mmap)
    
            assert self.evecs_cond_first.dtype == torch.float32, f'evecs_cond_first dtype: {self.evecs_cond_first.dtype}'
            assert self.evecs_cond_second.dtype == torch.float32, f'evecs_cond_second dtype: {self.evecs_cond_second.dtype}'
        
        
    
    def __len__(self):
        return len(self.fmaps)
    
    def __getitem__(self, idx):
        fmap = self.fmaps[idx]
        
        fmap = self.fmaps[idx].unsqueeze(0)      
        # fmap = fmap.reshape(1, self.fmap_dim, self.fmap_dim).float()
                
        # normalize to [0, 1] and to [-1, 1]
        if self.fmap_input_type == 'abs':
            fmap = fmap / fmap.max()
            fmap = fmap * 2 - 1 
        
        conditioning = torch.tensor([])
        
        if 'evals' in self.conditioning_types:
            eval = self.evals[idx].unsqueeze(0)
            eval = torch.diag_embed(eval)
            conditioning = torch.cat((conditioning, eval), 0)
        
        if 'evals_inv' in self.conditioning_types:
            eval_inv = 1 / self.evals[idx].unsqueeze(0)
            # replace elements > 1 with 1
            eval_inv[eval_inv > 1] = 1
            eval_inv = torch.diag_embed(eval_inv)
            conditioning = torch.cat((conditioning, eval_inv), 0)
        
        if 'evecs' in self.conditioning_types:
            evecs_cond_first = self.evecs_cond_first[idx].unsqueeze(0)            
            evecs_cond_second = self.evecs_cond_second[idx].unsqueeze(0)
            
            # evecs_cond_first = self.evecs_cond_first[idx]
            # evecs_cond_first = evecs_cond_first.reshape(1, self.fmap_dim, self.fmap_dim).float()
            
            # evecs_cond_second = self.evecs_cond_second[idx]
            # evecs_cond_second = evecs_cond_second.reshape(1, self.fmap_dim, self.fmap_dim).float()
            
            evecs = torch.cat((evecs_cond_first, evecs_cond_second), 0)
            conditioning = torch.cat((conditioning, evecs), 0)

        conditioning = conditioning.contiguous()
        
        return fmap, conditioning
    