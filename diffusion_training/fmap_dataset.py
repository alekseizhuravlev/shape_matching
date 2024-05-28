import torch
import numpy as np

class FMapDataset:
    def __init__(self, fmaps, evals):
        self.fmaps = fmaps
        
        fmap_dim = np.sqrt(fmaps.shape[1])
        self.fmaps = torch.tensor(self.fmaps, dtype=torch.float32).reshape(len(self.fmaps), fmap_dim, fmap_dim)
        self.fmaps = self.fmaps.abs()
        
        self.evals = evals
        self.evals = torch.tensor(self.evals, dtype=torch.float32)
    
    def __len__(self):
        return len(self.evals)
    
    def __getitem__(self, idx):
        fmap = self.fmaps[idx].unsqueeze(0)
        
        # normalize to [0, 1] and to [-1, 1]
        fmap = fmap / fmap.max()
        fmap = fmap * 2 - 1 
        
        eval = self.evals[idx].unsqueeze(0)
        
        # pad with 2 zeros
        # fmap = F.pad(fmap, (0, 2, 0, 2))
        # pad with 2 zeros to the right
        # eval = F.pad(eval, (0, 2))
        
        return fmap, eval
    
    
def create_train_test_loader(fmaps_file, evals_file, train_fraction, batch_size):
    # fmaps_full = np.loadtxt('/home/s94zalek/shape_matching/data/SURREAL_full/fmaps/fmaps_125_125_250_0_28.txt')
    # evals_full = np.loadtxt('/home/s94zalek/shape_matching/data/SURREAL_full/evals/evals_125_125_250_0_28.txt')
        
    fmaps_full = np.loadtxt(fmaps_file)
    evals_full = np.loadtxt(evals_file)
        
    train_indices = np.random.choice(len(fmaps_full), int(train_fraction * len(fmaps_full)), replace=False)
    test_indices = np.array([i for i in range(len(fmaps_full)) if i not in train_indices])

    print(f'Number of training samples: {len(train_indices)}, number of test samples: {len(test_indices)}')
        
    train_dataset = FMapDataset(fmaps_full[train_indices], evals_full[train_indices])
    test_dataset = FMapDataset(fmaps_full[test_indices], evals_full[test_indices])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Fmap shape: {train_dataset[10][0].shape}, eval shape: {train_dataset[10][1].shape}')
    
    return train_dataset, train_dataloader, test_dataset, test_dataloader