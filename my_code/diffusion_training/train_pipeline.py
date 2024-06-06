import torch
import numpy as np
import os
import shutil
from tqdm import tqdm
import yaml
from collections import OrderedDict

import sys
sys.path.append('/home/s94zalek/shape_matching')

# datasets
from my_code.datasets.surreal_cached_train_dataset import SurrealTrainDataset
from my_code.datasets.surreal_cached_test_dataset import SurrealTestDataset

# models
from my_code.models.diag_conditional import DiagConditionedUnet
from diffusers import DDPMScheduler

# training / evaluation
from torch.utils.tensorboard import SummaryWriter
from my_code.diffusion_training.train_model import train_epoch

from my_code.datasets.surreal_dataset import SingleSurrealDataset


# load config file

# load the dataset

# load the model
# possibly load the checkpoint

# load tensorboard

# start training

# training step
    # save the metrics to tensorboard
    # save the checkpoint

# validation step
    # save the metrics to tensorboard

# final validation step
# final save of the model


if __name__ == '__main__':
    
    # configuration
    config = {
        'experiment_name': 'test_noCache',
        'dataset_name': 'dataset_158_158_316_0_32_93',
        
        'n_epochs': 30,
        'validate_every': 10,
        'checkpoint_every': 10,
        'batch_size': 128,
        
        'model_params': {
            'sample_size': 32,
            'in_channels': 2,
            'out_channels': 1,
            'layers_per_block': 2,
            'block_out_channels': (32, 64, 64),
            'down_block_types': (
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            'up_block_types': (
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        }
    }   
    
    # experiment setup
    experiment_folder = f'/home/s94zalek/shape_matching/my_code/experiments/{config["experiment_name"]}'
    shutil.rmtree(experiment_folder, ignore_errors=True)
    os.makedirs(experiment_folder, exist_ok=True)
    os.makedirs(f'{experiment_folder}/checkpoints', exist_ok=True)
    
    # save the config file
    with open(f'{experiment_folder}/config.yaml', 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    
    ### Train and test datasets with dataloaders
    # dataset_base_folder = '/home/s94zalek/shape_matching/data/SURREAL_full/full_datasets'
    # train_dataset = SurrealTrainDataset(f'{dataset_base_folder}/{config["dataset_name"]}/train')
    # test_dataset = SurrealTestDataset(f'{dataset_base_folder}/{config["dataset_name"]}/test')
    
    train_dataset = SingleSurrealDataset(
        n_body_types_male=256,
        n_body_types_female=256,
        n_poses_straight=512 - 32,
        n_poses_bent=32,
        num_evecs=32,
        use_same_poses_male_female=False
    )
    test_dataset = SingleSurrealDataset(
        n_body_types_male=64,
        n_body_types_female=64,
        n_poses_straight=100,
        n_poses_bent=28,
        num_evecs=32,
        use_same_poses_male_female=False
    )
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    print(f'Number of training samples: {len(train_dataset)}, number of test samples: {len(test_dataset)}')
    print(f'Fmap shape: {train_dataset[10][0].shape}, eval shape: {train_dataset[10][1].shape}')
        
        
    ### Model
    # pass config["model_params"] to the model
    model = DiagConditionedUnet(config["model_params"]).to('cuda')
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                                clip_sample=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    tb_writer = SummaryWriter(log_dir=experiment_folder)
    
    
    ### Training
    train_iterator = tqdm(range(config["n_epochs"]))
    for epoch in train_iterator:
        
        # training step
        model, losses = train_epoch(model, is_unconditional=False,
                                    train_dataloader=train_dataloader, noise_scheduler=noise_scheduler,
                                    opt=opt, loss_fn=loss_fn)
        # save the losses to tensorboard
        for i, loss_value in enumerate(losses):
            tb_writer.add_scalar(f'loss/train', loss_value, epoch * len(train_dataloader) + i)
            
        train_iterator.set_description(f'Epoch {epoch}, loss: {sum(losses[-100:])/100:.4f}')
            
            
            
            
        # validation step
        ## TODO
        
        
        # save the model checkpoint
        if epoch > 0 and (epoch % config["checkpoint_every"] == 0 or epoch == config["n_epochs"] - 1):
            torch.save(model.state_dict(), f'{experiment_folder}/checkpoints/checkpoint_{epoch}.pt')
        
        
        
    
    
    

    
    
    