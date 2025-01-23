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
from diffusers.optimization import get_cosine_schedule_with_warmup

# training / evaluation
from torch.utils.tensorboard import SummaryWriter
from my_code.diffusion_training_sign_corr.train_model import train_epoch
from my_code.diffusion_training_sign_corr.validate_model import validate_epoch

import my_code.diffusion_training_sign_corr.data_loading as data_loading

from my_code.datasets.surreal_cached_train_dataset import SurrealTrainDataset
import networks.diffusion_network as diffusion_network

from accelerate import Accelerator
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--dataset_name', type=str)
    
    parser.add_argument('--fmap_direction', type=str)
    parser.add_argument('--sample_size', type=int)
    
    parser.add_argument('--block_out_channels', type=str)
    parser.add_argument('--down_block_types', type=str)
    parser.add_argument('--up_block_types', type=str)
    
    args = parser.parse_args()
    
    return args


def main():
    
    args = parse_args()
    
    
    
    # configuration
    config = {
        'experiment_name': args.experiment_name,
        'experiment_base_dir': '/lustre/mlnvme/data/s94zalek_hpc-shape_matching/ddpm_checkpoints',
        'accelerate': True,
        
        'dataset_base_dir': '/tmp',
        'dataset_name': args.dataset_name,
        
        'fmap_direction': args.fmap_direction,
        'fmap_type': 'orig',
        'conditioning_types': {'evecs'},
        
        'n_epochs': 100,
        'validate_every': 10,
        'checkpoint_every': 10,
        
        'batch_size': 64,
        'eval_batch_size': 64,
        
        'model_params': {
            'sample_size': args.sample_size,
            'in_channels': 3,
            'out_channels': 1,
            'layers_per_block': 2,
            'block_out_channels': tuple(map(int, args.block_out_channels.split(','))),
            'down_block_types': tuple(args.down_block_types.split(',')),
            'up_block_types': tuple(args.up_block_types.split(',')),
        },
    }   
    
    # experiment setup
    # experiment_folder = f'/home/s94zalek_hpc/shape_matching/my_code/experiments/ddpm/{config["experiment_name"]}'
    experiment_folder = f'{config["experiment_base_dir"]}/{config["experiment_name"]}'
    
    
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
        os.makedirs(f'{experiment_folder}/checkpoints_state', exist_ok=True)
        os.makedirs(f'{experiment_folder}/checkpoints_state_dicts', exist_ok=True)
        
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
        f'{config["dataset_base_dir"]}/{config["dataset_name"]}/train',
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
    
    # print all args
    print('All args:')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
        
    print(f'Fmap shape: {dataset_train[10][0].shape}, ', 
          f'conditioning shape: {dataset_train[10][1].shape}')
        
                
    ### Model
    model = DiagConditionedUnet(config["model_params"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    #     opt, start_factor=1, end_factor=0.1, 
    #     total_iters=config["n_epochs"] * len(dataloader_train)
    #     )
    
    # use cosine annealing
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     opt, T_max=config["n_epochs"] * len(dataloader_train)
    #     )
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=len(dataloader_train) // 2,
        num_training_steps=config["n_epochs"] * len(dataloader_train),
    )
    
    ####################################################
    # !!!!!! dropping this will cause 
    # UserWarning: Grad strides do not match bucket view strides. 
    # grad.sizes() = [128, 256, 1, 1], strides() = [256, 1, 256, 256]
    # bucket_view.sizes() = [128, 256, 1, 1], strides() = [256, 1, 1, 1]
    
    model.to(memory_format=torch.channels_last)
    
    ####################################################
    
    model, opt, dataloader_train, lr_scheduler = accelerator.prepare(
        model, opt, dataloader_train, lr_scheduler
    )
    
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                                clip_sample=True)
    loss_fn = torch.nn.MSELoss()
    
    accelerator.init_trackers(config["experiment_name"])
    
    
    # print(model.state_dict())
    
    # exit(0)
    
    
    ### Training
    train_iterator = tqdm(range(config["n_epochs"]), disable=not accelerator.is_local_main_process)
    for epoch in train_iterator: 
        
        # training step
        model, losses = train_epoch(model, is_unconditional=False,
                                    train_dataloader=dataloader_train,
                                    noise_scheduler=noise_scheduler,
                                    opt=opt, loss_fn=loss_fn,
                                    accelerator=accelerator,
                                    lr_scheduler=lr_scheduler)
        
        # save the losses to tensorboard
        # for i, loss_value in enumerate(losses):
        #     accelerator.log({f'loss/train': loss_value}, step = epoch * len(dataloader_train) + i)
        
        # only log the last loss
        accelerator.log({f'loss/train': losses[-1]}, step = epoch * len(dataloader_train))
            
            
        train_iterator.set_description(f'Epoch {epoch}, loss: {sum(losses[-100:])/100:.4f}')
        
                    
        # save the model checkpoint
        if epoch > 0 and (epoch % config["checkpoint_every"] == 0 or epoch == config["n_epochs"] - 1):
            
            accelerator.wait_for_everyone()
            accelerator.save_model(model, f'{experiment_folder}/checkpoints/epoch_{epoch}')
            
            # accelerator.save_state(f'{experiment_folder}/checkpoints_state/epoch_{epoch}')
            
            # state_dict = accelerator.get_state_dict(model, unwrap=True)
            # torch.save(state_dict, f'{experiment_folder}/checkpoints_state_dicts/epoch_{epoch}.pt')
                


    accelerator.end_training()
        
if __name__ == '__main__':
    main()        
    
    

    
    