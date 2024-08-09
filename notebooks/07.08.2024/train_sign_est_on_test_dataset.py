import torch
import numpy as np
import matplotlib.pyplot as plt

import trimesh

scene = trimesh.Scene()

from my_code.datasets.surreal_dataset_3dc import TemplateSurrealDataset3DC
import torch
from tqdm import tqdm
import time
import my_code.diffusion_training_sign_corr.data_loading as data_loading
import networks.diffusion_network as diffusion_network
from tqdm import tqdm
import my_code.sign_canonicalization.training as sign_training
import pandas as pd
import os



if __name__ == '__main__':
    
    base_dir = '/home/s94zalek_hpc/shape_matching/my_code/experiments'
    
    experiment_name = 'signCorr_FAUST_r'
    
    experiment_dir = f'{base_dir}/{experiment_name}'
    os.makedirs(experiment_dir, exist_ok=True)
    

    train_dataset = data_loading.get_val_dataset(
        'FAUST_r', 'train', 128, canonicalize_fmap=None
        )[1]


    condition_dim = 0
    start_dim = 0

    feature_dim = 32
    evecs_per_support = 4


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = diffusion_network.DiffusionNet(
        in_channels=feature_dim,
        out_channels=feature_dim // evecs_per_support,
        cache_dir=None,
        input_type='wks',
        k_eig=128,
        n_block=6
        ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1, end_factor=0.1, 
        total_iters=50000)


    tqdm._instances.clear()

    loss_fn = torch.nn.MSELoss()
    losses = torch.tensor([])
    train_iterator = tqdm(range(20000))     
            
    curr_iter = 0
    for epoch in range(len(train_iterator) // len(train_dataset)):
        
        # train_shapes_shuffled = train_shapes.copy()
        # np.random.shuffle(train_shapes)
        
        
        for curr_idx in range(len(train_dataset)):

            ##############################################
            # Select a shape
            ##############################################
            # curr_idx = np.random.randint(0, len(train_shapes))
        
            train_shape = train_dataset[curr_idx]['second']

            # train_shape = double_shape['second']
            verts = train_shape['verts'].unsqueeze(0).to(device)
            faces = train_shape['faces'].unsqueeze(0).to(device)    

            evecs_orig = train_shape['evecs'].unsqueeze(0)[:, :, start_dim:start_dim+feature_dim].to(device)
            
            mass_mat = torch.diag_embed(
                torch.ones_like(train_shape['mass'].unsqueeze(0))
                ).to(device)

            ##############################################
            # Set the signs on shape 0
            ##############################################

            # create a random combilation of +1 and -1, length = feature_dim
            sign_gt_0 = torch.randint(0, 2, (feature_dim,)).float().to(device)
            
            sign_gt_0[sign_gt_0 == 0] = -1
            sign_gt_0 = sign_gt_0.float().unsqueeze(0)

            # multiply evecs [6890 x 16] by sign_flip [16]
            evecs_flip_0 = evecs_orig * sign_gt_0
            
            # predict the sign change
            sign_pred_0 = sign_training.predict_sign_change(
                net, verts, faces, evecs_flip_0, 
                mass_mat=mass_mat, input_type=net.input_type,
                
                mass=train_shape['mass'].unsqueeze(0), L=train_shape['L'].unsqueeze(0),
                evals=train_shape['evals'].unsqueeze(0), evecs=train_shape['evecs'].unsqueeze(0),
                gradX=train_shape['gradX'].unsqueeze(0), gradY=train_shape['gradY'].unsqueeze(0)
                )[0]
            
            ##############################################
            # Set the signs on shape 1
            ##############################################
            
            # create a random combilation of +1 and -1, length = feature_dim
            sign_gt_1 = torch.randint(0, 2, (feature_dim,)).float().to(device)
            
            sign_gt_1[sign_gt_1 == 0] = -1
            sign_gt_1 = sign_gt_1.float().unsqueeze(0)
            
            # multiply evecs [6890 x 16] by sign_flip [16]
            evecs_flip_1 = evecs_orig * sign_gt_1
            
            # predict the sign change
            sign_pred_1 = sign_training.predict_sign_change(
                net, verts, faces, evecs_flip_1, 
                mass_mat=mass_mat, input_type=net.input_type,
                
                mass=train_shape['mass'].unsqueeze(0), L=train_shape['L'].unsqueeze(0),
                evals=train_shape['evals'].unsqueeze(0), evecs=train_shape['evecs'].unsqueeze(0),
                gradX=train_shape['gradX'].unsqueeze(0), gradY=train_shape['gradY'].unsqueeze(0)
                )[0]
            
            ##############################################
            # Calculate the loss
            ##############################################
            
            # calculate the ground truth sign difference
            sign_diff_gt = sign_gt_1 * sign_gt_0
            
            # calculate the sign difference between predicted evecs
            sign_diff_pred = sign_pred_1 * sign_pred_0
            
            # calculate the loss
            loss = loss_fn(
                sign_diff_pred.reshape(sign_diff_pred.shape[0], -1),
                sign_diff_gt.reshape(sign_diff_gt.shape[0], -1)
                )

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            
            losses = torch.cat([losses, torch.tensor([loss.item()])])
            
            # print mean of last 10 losses
            train_iterator.set_description(f'loss={torch.mean(losses[-10:]):.3f}')
            
            # plot the losses every 1000 iterations
            if curr_iter > 0 and curr_iter % (len(train_iterator) // 10) == 0:
                pd.Series(losses.numpy()).rolling(10).mean().plot()
                plt.yscale('log')
                plt.show()
                
                plt.savefig(f'{experiment_dir}/losses_{curr_iter}.png')
                plt.close()
                
                torch.save(
                    net.state_dict(),
                    f'{experiment_dir}/{curr_iter}.pth'
                    )
                
            curr_iter += 1
            train_iterator.update(1)
            
    torch.save(
    net.state_dict(),
    f'{experiment_dir}/{curr_iter}.pth'
    )
            
            