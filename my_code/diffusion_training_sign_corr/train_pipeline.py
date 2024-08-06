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


def get_val_datasets(config):  
    return [
    {
        'name': 'train-FAUST-original',
        'dataset': data_loading.get_val_dataset('FAUST_orig', 'train', config["model_params"]["sample_size"])[1],
    },
    {
        'name': 'test-FAUST-original',
        'dataset': data_loading.get_val_dataset('FAUST_orig', 'test', config["model_params"]["sample_size"])[1],
    },
    {
        'name': 'test-FAUST-r',
        'dataset': data_loading.get_val_dataset('FAUST_r', 'test', config["model_params"]["sample_size"])[1],
    },
    {
        'name': 'test-FAUST-a',
        'dataset': data_loading.get_val_dataset('FAUST_a', 'test', config["model_params"]["sample_size"])[1],
    },
    {
        'name': 'test-SURREAL',
        'dataset': data_loading.get_val_dataset('SURREAL', 'test', config["model_params"]["sample_size"])[1],
    }
    ]



if __name__ == '__main__':
    
    # configuration
    config = {
        'experiment_name': 'test_signCorr_withAug_evalsInvEvecs_32',
        
        'dataset_name': 'dataset_SURREAL_train_withAug_productSuppCond_32',
        'fmap_type': 'orig',
        'conditioning_types': {'evals_inv', 'evecs'},
        
        
        'sign_net_path': '/home/s94zalek_hpc/shape_matching/my_code/experiments/sign_estimator_no_aug/40000.pth',
        'net_input_type': 'wks',
        'evecs_per_support': 4,
        
        'n_epochs': 100,
        'validate_every': 5,
        'checkpoint_every': 5,
        
        'batch_size': 128,
        'eval_batch_size': 128,
        
        'model_params': {
            'sample_size': 32,
            'in_channels': 4,
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
    # shutil.rmtree(experiment_folder, ignore_errors=True)
    os.makedirs(experiment_folder)
    os.makedirs(f'{experiment_folder}/checkpoints')

    # save the config file
    with open(f'{experiment_folder}/config.yaml', 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    
    ### Train dataset with dataloader
    dataset_train = SurrealTrainDataset(
        f'data/SURREAL_full/full_datasets/{config["dataset_name"]}/train',
        fmap_input_type=config["fmap_type"],
        conditioning_types=config["conditioning_types"]
        )
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True)
       
    # validation datasets
    # val_datasets_payload = get_val_datasets(config)
    val_datasets_payload = []
    

    print(f'Number of training samples: {len(dataset_train)}')
    for val_dataset in val_datasets_payload:
        print(f'Number of {val_dataset["name"]} samples: {len(val_dataset["dataset"])}')
        
    print(f'Fmap shape: {dataset_train[10][0].shape}, ', 
          f'conditioning shape: {dataset_train[10][1].shape}')
        
        
    ### Model
    model = DiagConditionedUnet(config["model_params"]).to('cuda')
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                                clip_sample=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    tb_writer = SummaryWriter(log_dir=experiment_folder)
    
    
    ### Sign correction network
    sign_corr_net = diffusion_network.DiffusionNet(
        in_channels=config["model_params"]["sample_size"],
        out_channels=config["model_params"]["sample_size"] // config["evecs_per_support"],
        cache_dir=None,
        input_type=config["net_input_type"],
        k_eig=128,
        n_block=6
        ).to('cuda')
    sign_corr_net.load_state_dict(torch.load(config["sign_net_path"]))
    sign_corr_net.eval()
    
    
    ### Training
    train_iterator = tqdm(range(config["n_epochs"]))
    for epoch in train_iterator:
        
        # validation step
        # if epoch % config["validate_every"] == 0 or epoch == config["n_epochs"] - 1 or epoch == 0:
        #     with torch.no_grad():
        #         # iterate over the validation datasets
        #         for val_payload in val_datasets_payload:
        #             model, metrics_payload, figures_payload = validate_epoch(
        #                 model=model,
        #                 noise_scheduler=noise_scheduler,
        #                 test_dataset=val_payload["dataset"],
        #                 sign_corr_net=sign_corr_net
        #             ) 
                    
        #             if epoch > 0:
        #                 # save the metrics to tensorboard                
        #                 for k, v in metrics_payload.items():
        #                     tb_writer.add_scalar(f'{k}/{val_payload["name"]}', v, epoch)
        #                 for k, figure in figures_payload.items():
        #                     tb_writer.add_figure(f'{k}/{val_payload["name"]}', figure, epoch)
                    
        
        # training step
        model, losses = train_epoch(model, is_unconditional=False,
                                    train_dataloader=dataloader_train,
                                    noise_scheduler=noise_scheduler,
                                    opt=opt, loss_fn=loss_fn)
        
        # save the losses to tensorboard
        for i, loss_value in enumerate(losses):
            tb_writer.add_scalar(f'loss/train',
                                 loss_value, epoch * len(dataloader_train) + i
                                 )
            
        train_iterator.set_description(f'Epoch {epoch}, loss: {sum(losses[-100:])/100:.4f}')
            
        
                    
        # save the model checkpoint
        if epoch > 0 and (epoch % config["checkpoint_every"] == 0 or epoch == config["n_epochs"] - 1):
            torch.save(model.state_dict(), f'{experiment_folder}/checkpoints/checkpoint_{epoch}.pt')
        
        
        
    
    
    

    
    
    