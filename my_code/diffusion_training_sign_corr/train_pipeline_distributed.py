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
from my_code.diffusion_training_sign_corr.train_model import train_epoch
from my_code.diffusion_training_sign_corr.validate_model import validate_epoch

import my_code.diffusion_training_sign_corr.data_loading as data_loading

from my_code.datasets.surreal_cached_train_dataset import SurrealTrainDataset
import networks.diffusion_network as diffusion_network

from accelerate import Accelerator


def main():
    
    # configuration
    config = {
        'experiment_name': 'pair_5_xy_distributed',
        'accelerate': True,
        
        'dataset_base_dir': '/tmp',
        'dataset_name': 'pair_5_augShapes_signNet_remeshed_mass_6b_1ev_10_0.2_0.8',
        
        'fmap_direction': 'xy',
        'fmap_type': 'orig',
        'conditioning_types': {'evecs'},
        
        'n_epochs': 100,
        'validate_every': 5,
        'checkpoint_every': 1,
        
        'batch_size': 128,
        'eval_batch_size': 128,
        
        'model_params': {
            'sample_size': 32,
            'in_channels': 3,
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
        },
    }   
    
    # experiment setup
    experiment_folder = f'/home/{user_name}/shape_matching/my_code/experiments/ddpm/{config["experiment_name"]}'
    # shutil.rmtree(experiment_folder, ignore_errors=True)
    
    
    # if os.path.exists(experiment_folder):
    #     # make a prompt to remove the directory
    #     print(f'{experiment_folder} already exists')
    #     print('Press y to remove, any other key to exit')
        
    #     user_input = input()
    #     if user_input == 'y':
    #         print('Removing', experiment_folder)
    #         shutil.rmtree(experiment_folder)
    

    # Accelerator
    accelerator = Accelerator(project_dir=experiment_folder,
                              log_with='tensorboard')
    device = accelerator.device
    
    if accelerator.is_local_main_process:
        
        os.makedirs(experiment_folder, exist_ok=True)
        os.makedirs(f'{experiment_folder}/checkpoints', exist_ok=True)
        
        # sign net config
        with open(f'{config["dataset_base_dir"]}/{config["dataset_name"]}/config.yaml', 'r') as f:
            sign_net_config = yaml.load(f, Loader=yaml.FullLoader)
            
        # add the sign net config to the main config
        config['sign_net'] = sign_net_config

        # save the config file
        with open(f'{experiment_folder}/config.yaml', 'w') as f:
            yaml.dump(config, f, sort_keys=False)
    
    
    ### Train dataset with dataloader
    dataset_train = SurrealTrainDataset(
        # f'data/SURREAL_full/full_datasets/{config["dataset_name"]}/train',
        f'{config["dataset_base_dir"]}/{config["dataset_name"]}',
        fmap_direction=config["fmap_direction"],
        fmap_input_type=config["fmap_type"],
        conditioning_types=config["conditioning_types"],
        mmap=True
        )
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True)
       
    # validation datasets
    # val_datasets_payload = get_val_datasets(config)
    # val_datasets_payload = []
    

    # print(f'Number of training samples: {len(dataset_train)}')
    # for val_dataset in val_datasets_payload:
    #     print(f'Number of {val_dataset["name"]} samples: {len(val_dataset["dataset"])}')
        
    print(f'Fmap shape: {dataset_train[10][0].shape}, ', 
          f'conditioning shape: {dataset_train[10][1].shape}')
        
                
    ### Model
    model = DiagConditionedUnet(config["model_params"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model, opt, dataloader_train = accelerator.prepare(
        model, opt, dataloader_train
    )
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                                clip_sample=True)
    loss_fn = torch.nn.MSELoss()
    
    accelerator.init_trackers(config["experiment_name"])
    
    ### Training
    train_iterator = tqdm(range(config["n_epochs"]), disable=not accelerator.is_local_main_process)
    for epoch in train_iterator: 
        
        # training step
        model, losses = train_epoch(model, is_unconditional=False,
                                    train_dataloader=dataloader_train,
                                    noise_scheduler=noise_scheduler,
                                    opt=opt, loss_fn=loss_fn,
                                    accelerator=accelerator)
        
        # save the losses to tensorboard
        for i, loss_value in enumerate(losses):
            accelerator.log({f'loss/train': loss_value}, step = epoch * len(dataloader_train) + i)
            
            
        train_iterator.set_description(f'Epoch {epoch}, loss: {sum(losses[-100:])/100:.4f}')
        
                    
        # save the model checkpoint
        if epoch > 0 and (epoch % config["checkpoint_every"] == 0 or epoch == config["n_epochs"] - 1):
                accelerator.wait_for_everyone()
                accelerator.save_model(model, f'{experiment_folder}/checkpoints/epoch_{epoch}')

    accelerator.end_training()
        
if __name__ == '__main__':
    main()        
    
    
    

    
    
    