import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def sample_dataloader(model, test_loader, noise_scheduler):

    device = model.device()
    
    x_sampled_list = []
    
    print('Sampling test loader, device =', device)
    for batch in tqdm(test_loader, desc='Sampling test loader...'):
        
        # print(batch)
        x_gt, y = batch['second']['C_gt_xy'], batch['second']['evals']  

        # Prepare random x to start from, plus some desired labels y
        x_sampled = torch.rand_like(x_gt).to(device)  
        y = y.to(device)    
            
        # Sampling loop
        for i, t in tqdm(list(enumerate(noise_scheduler.timesteps)), total=noise_scheduler.config.num_train_timesteps,
                         desc='Denoising...'):

            # Get model pred
            with torch.no_grad():
                residual = model(x_sampled, t,
                                    conditioning=y
                                    ).sample

            # Update sample with step
            x_sampled = noise_scheduler.step(residual, t, x_sampled).prev_sample

        x_sampled_list.append(x_sampled.cpu())     
        
    x_sampled_list = torch.cat(x_sampled_list, dim=0)
        
    return x_sampled_list
        
        

def sample_dataset(model, test_dataset, noise_scheduler):

    device = model.device()
    
    print('Sampling test dataset, device =', device)
    
    # get ground truth fmap and evals from test set
    x_gt = []
    y = []
    for i in tqdm(range(len(test_dataset)), desc='Gathering evals and fmaps...'):
        x_gt.append(test_dataset[i]['second']['C_gt_xy'])
        y.append(test_dataset[i]['second']['evals'])
    x_gt = torch.stack(x_gt)
    y = torch.stack(y)
    

    # Prepare random x to start from, plus some desired labels y
    x_sampled = torch.rand_like(x_gt).to(device)  
    y = y.to(device)    
        
    # Sampling loop
    for i, t in tqdm(list(enumerate(noise_scheduler.timesteps)), total=noise_scheduler.config.num_train_timesteps,
                        desc='Denoising...'):

        # Get model pred
        with torch.no_grad():
            residual = model(x_sampled, t,
                                conditioning=y
                                ).sample

        # Update sample with step
        x_sampled = noise_scheduler.step(residual, t, x_sampled).prev_sample

    return x_sampled.cpu()
        
        
        
        
if __name__ == '__main__':
    
    import sys
    sys.path.append('/home/s94zalek/shape_matching')

    # datasets
    from my_code.datasets.surreal_cached_train_dataset import SurrealTrainDataset
    from shape_matching.my_code.datasets.surreal_legacy.surreal_cached_test_dataset import SurrealTestDataset

    # models
    from my_code.models.diag_conditional import DiagConditionedUnet
    from diffusers import DDPMScheduler
    
    import yaml
    import torch
    
    from diffusers import DDPMScheduler

    exp_dir = '/home/s94zalek/shape_matching/my_code/experiments/test_32'

    with open(exp_dir + '/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_base_folder = '/home/s94zalek/shape_matching/data/SURREAL_full/full_datasets'
    test_dataset = SurrealTestDataset(f'{dataset_base_folder}/{config["dataset_name"]}/test')

    model = DiagConditionedUnet(config["model_params"]).to('cuda')

    # load checkpoint_29.pt
    model.load_state_dict(torch.load(exp_dir + '/checkpoint_29.pt'))
    model = model.to('cuda')
    

    batch_size = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                                    clip_sample=True)

    x_sampled = sample(model, test_loader, noise_scheduler)

    with open(exp_dir + '/x_sampled_29.pt', 'wb') as f:
        torch.save(x_sampled, f)