import torch
import numpy as np


class SurrealTrainDataset(torch.utils.data.Dataset):
    def __init__(self, base_folder, input_type):
        super(SurrealTrainDataset, self).__init__()
        
        self.input_type = input_type
        
        
        self.fmaps = np.loadtxt(f'{base_folder}/C_gt_xy.txt')       
    
        fmap_dim = int(np.sqrt(self.fmaps.shape[1]))
        print('Train dataset, functional map dimension:', fmap_dim)
        self.fmaps = torch.tensor(self.fmaps, dtype=torch.float32).reshape(len(self.fmaps), fmap_dim, fmap_dim)
        
        if self.input_type == 'abs':
            self.fmaps = self.fmaps.abs()
        
        self.evals = np.loadtxt(f'{base_folder}/evals.txt')
        self.evals = torch.tensor(self.evals, dtype=torch.float32)
    
    def __len__(self):
        return len(self.evals)
    
    def __getitem__(self, idx):
        fmap = self.fmaps[idx].unsqueeze(0)
        # fmap = self.fmaps[idx]
        
        # normalize to [0, 1] and to [-1, 1]
        if self.input_type == 'abs':
            fmap = fmap / fmap.max()
            fmap = fmap * 2 - 1 
        
        eval = self.evals[idx].unsqueeze(0)
        # eval = self.evals[idx]
        
        # pad with 2 zeros
        # fmap = F.pad(fmap, (0, 2, 0, 2))
        # pad with 2 zeros to the right
        # eval = F.pad(eval, (0, 2))
        
        return fmap, eval
    