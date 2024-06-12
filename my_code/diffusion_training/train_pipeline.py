import torch
import numpy as np
import os
import shutil
from tqdm import tqdm
import yaml

import sys
import os
curr_dir = os.getcwd()
if 's94zalek_hpc' in curr_dir:
    user_name = 's94zalek_hpc'
else:
    user_name = 's94zalek'
sys.path.append(f'/home/{user_name}/shape_matching')

# models
from my_code.models.diag_conditional import DiagConditionedUnet
from diffusers import DDPMScheduler

# training / evaluation
from torch.utils.tensorboard import SummaryWriter
from my_code.diffusion_training.train_model import train_epoch
from my_code.diffusion_training.validate_model import validate_epoch

from data_loading import get_datasets


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
        'experiment_name': 'test_3DCoded_100_scaledFaust',
        'dataset_name': 'dataset_3dc_32',
        
        'n_epochs': 100,
        'validate_every': 5,
        'checkpoint_every': 5,
        
        'batch_size': 128,
        'eval_batch_size': 64,
        
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
    experiment_folder = f'/home/{user_name}/shape_matching/my_code/experiments/{config["experiment_name"]}'
    shutil.rmtree(experiment_folder, ignore_errors=True)
    os.makedirs(experiment_folder, exist_ok=True)
    os.makedirs(f'{experiment_folder}/checkpoints', exist_ok=True)
    # os.makedirs(f'{experiment_folder}/cache_lb', exist_ok=True)
    
    # save the config file
    with open(f'{experiment_folder}/config.yaml', 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    
    ### Train and test datasets with dataloaders
    datasets_payload = get_datasets(config)

    print(f'Number of training samples: {len(datasets_payload["train"]["dataset"])}')
    for val_dataset in datasets_payload["val"]:
        print(f'Number of {val_dataset["name"]} samples: {len(val_dataset["dataset"])}')
        
    print(f'Fmap shape: {datasets_payload["train"]["dataset"][10][0].shape}, ', 
          f'eval shape: {datasets_payload["train"]["dataset"][10][1].shape}')
        
        
    ### Model
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
                                    train_dataloader=datasets_payload["train"]["dataloader"],
                                    noise_scheduler=noise_scheduler,
                                    opt=opt, loss_fn=loss_fn)
        
        # save the losses to tensorboard
        for i, loss_value in enumerate(losses):
            tb_writer.add_scalar(f'loss/train',
                                 loss_value, epoch * len(datasets_payload["train"]["dataloader"]) + i
                                 )
            
        train_iterator.set_description(f'Epoch {epoch}, loss: {sum(losses[-100:])/100:.4f}')
            
            
        # validation step
        if epoch > -1 and (epoch % config["validate_every"] == 0 or epoch == config["n_epochs"] - 1 or epoch == 0):
            with torch.no_grad():
                # iterate over the validation datasets
                for val_payload in datasets_payload["val"]:
                    model, metrics_payload, figures_payload = validate_epoch(
                        model=model,
                        noise_scheduler=noise_scheduler,
                        val_payload=val_payload
                    ) 
                    
                    # save the metrics to tensorboard                
                    for k, v in metrics_payload.items():
                        tb_writer.add_scalar(f'{k}/{val_payload["name"]}', v, epoch)
                    for k, figure in figures_payload.items():
                        tb_writer.add_figure(f'{k}/{val_payload["name"]}', figure, epoch)
                    
                    
        # save the model checkpoint
        if epoch > 0 and (epoch % config["checkpoint_every"] == 0 or epoch == config["n_epochs"] - 1):
            torch.save(model.state_dict(), f'{experiment_folder}/checkpoints/checkpoint_{epoch}.pt')
        
        
        
    
    
    

    
    
    