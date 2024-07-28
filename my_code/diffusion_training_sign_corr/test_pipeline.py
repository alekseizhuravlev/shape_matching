import os
import sys
curr_dir = os.getcwd()
if 's94zalek_hpc' in curr_dir:
    user_name = 's94zalek_hpc'
else:
    user_name = 's94zalek'
sys.path.append(f'/home/{user_name}/shape_matching')

# datasets
import my_code.diffusion_training_sign_corr.sample_model as sample_model
import my_code.diffusion_training_sign_corr.evaluate_samples as evaluate_samples


# models
from my_code.models.diag_conditional import DiagConditionedUnet
from diffusers import DDPMScheduler

import yaml
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import my_code.datasets.template_dataset as template_dataset
import my_code.datasets.shape_dataset as shape_dataset

import my_code.diffusion_training_sign_corr.data_loading as data_loading



def get_subset(dataset, subset_fraction):
    
    # get n random samples
    n_samples = int(len(dataset) * subset_fraction)
    subset_indices = torch.randperm(len(dataset))[:n_samples]
    
    # return the subset
    return torch.utils.data.Subset(dataset, subset_indices), subset_indices


def plot_pck(metrics, title):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    thresholds = np.linspace(0., 0.1, 40)
    ax.plot(thresholds, torch.mean(metrics['pcks'], axis=0), 'r-',
            label=f'auc: {torch.mean(metrics["auc"]):.2f}')
    ax.set_xlim(0., 0.1)
    ax.set_ylim(0, 1)
    ax.set_xscale('linear')
    ax.set_xticks([0.025, 0.05, 0.075, 0.1])
    ax.grid()
    ax.legend()
    ax.set_title(title)
    return fig


def preprocess_metrics(metrics):
    metrics_payload = {}
    
    metrics_payload['auc'] = round(metrics['auc'].mean(dim=0).item(), 2)
    
    metrics_payload['geo_err_mean'] = round(metrics['geo_err_est'].mean().item() * 100, 1)
    metrics_payload['geo_err_ratio_mean'] = round(metrics['geo_err_ratio'].mean().item(), 2)
    metrics_payload['geo_err_ratio_median'] = round(metrics['geo_err_ratio'].median().item(), 2)
    metrics_payload['geo_err_ratio_max'] = round(metrics['geo_err_ratio'].max().item(), 2)
    metrics_payload['geo_err_ratio_min'] = round(metrics['geo_err_ratio'].min().item(), 2)
    
    metrics_payload['mse_mean'] = round(metrics['mse_abs'].mean().item(), 2)
    metrics_payload['mse_median'] = round(metrics['mse_abs'].median().item(), 2)
    metrics_payload['mse_max'] = round(metrics['mse_abs'].max().item(), 2)
    metrics_payload['mse_min'] = round(metrics['mse_abs'].min().item(), 2)
    
    return metrics_payload




if __name__ == '__main__':

    experiment_name = 'test_faceScaling_faustRA'
    checkpoint_name = 'checkpoint_60'
    subset_fraction = 100
    # dataset_name = 'FAUST_orig'
    dataset_name = 'SHREC19'


    ### config
    exp_base_folder = f'/home/{user_name}/shape_matching/my_code/experiments/{experiment_name}'
    with open(f'{exp_base_folder}/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    ### model
    model = DiagConditionedUnet(config["model_params"]).to('cuda')
    model.load_state_dict(torch.load(f"{exp_base_folder}/checkpoints/{checkpoint_name}.pt"))
    model = model.to('cuda')


    ### test dataset
    test_dataset = data_loading.get_val_dataset(
        dataset_name, 'train', config["model_params"]["sample_size"]
        )[1]

    # return the subset
    subset_indices = list(range(len(test_dataset))) #[10:]
    test_dataset = torch.utils.data.Subset(test_dataset, subset_indices)


    ### sample the model
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                                    clip_sample=True)
    x_sampled = sample_model.sample_dataset(model, test_dataset, noise_scheduler)    


    ### unnormalize the samples and assign gt signs
    x_gt = torch.stack([test_dataset[i]['second']['C_gt_xy'] for i in range(len(test_dataset))])

    fmap_sampled = []
    for i in range(len(x_sampled)):
        fmap_i = x_sampled[i].cpu()

        fmap_i = (fmap_i + 1) / 2
        
        # set the sign to 0 for elements with absolute value < 0.05
        sign_gt_i = torch.sign(x_gt[i])
        sign_gt_i[torch.abs(x_gt[i]) < 0.05] = 0
        fmap_i = fmap_i * sign_gt_i
        
        # fmap_i = fmap_i * torch.sign(x_gt[i])


        fmap_sampled.append(fmap_i)
    fmap_sampled = torch.stack(fmap_sampled)


    ### calculate metrics
    metrics = evaluate_samples.calculate_metrics(
        fmap_sampled,
        test_dataset
    )
    metrics_gt = evaluate_samples.calculate_metrics(
        x_gt,
        test_dataset
    )

    metrics_payload = preprocess_metrics(metrics)
    metrics_payload_gt = preprocess_metrics(metrics_gt)

    fig = plot_pck(metrics, metrics_gt, title=f"PCK on {dataset_name}_{subset_fraction}")

    ### print the metrics
    print(f"AUC mean: {metrics_payload['auc']}\n")
    print(f'GeoErr mean: {metrics_payload["geo_err_mean"]}\n')
    print(f'GeoErr median: {metrics_payload["geo_err_median"]}\n')

    print(f"GeoErr ratio mean: {metrics_payload['geo_err_ratio_mean']}")
    print(f"GeoErr ratio median: {metrics_payload['geo_err_ratio_median']}")
    print(f'GeoErr ratio max: {metrics_payload["geo_err_ratio_max"]}', f'min: {metrics_payload["geo_err_ratio_min"]}\n')
    print(f"MSE mean: {metrics_payload['mse_mean']}")
    print(f"MSE median: {metrics_payload['mse_median']}")
    print(f"MSE max: {metrics_payload['mse_max']}", f"min: {metrics_payload['mse_min']}")


    print(f"\n\nAUC mean_gt: {metrics_payload_gt['auc']}\n")
    print(f'GeoErr mean_gt: {metrics_payload_gt["geo_err_mean"]}\n')
    print(f'GeoErr median_gt: {metrics_payload_gt["geo_err_median"]}\n')

    print(f"GeoErr ratio mean_gt: {metrics_payload_gt['geo_err_ratio_mean']}")
    print(f"GeoErr ratio median_gt: {metrics_payload_gt['geo_err_ratio_median']}")
    print(f'GeoErr ratio max_gt: {metrics_payload_gt["geo_err_ratio_max"]}', f'min_gt: {metrics_payload_gt["geo_err_ratio_min"]}\n')
    print(f"MSE mean_gt: {metrics_payload_gt['mse_mean']}")
    print(f"MSE median_gt: {metrics_payload_gt['mse_median']}")
    print(f"MSE max_gt: {metrics_payload_gt['mse_max']}", f"min_gt: {metrics_payload_gt['mse_min']}")

    
    ### save the metrics and samples
    save_folder = f"{exp_base_folder}/evaluation"
    os.makedirs(save_folder, exist_ok=True)
    
    # metrics
    with open(f"{save_folder}/metrics_{dataset_name}_{subset_fraction}.yaml", 'w') as f:
        yaml.dump(metrics_payload, f, sort_keys=False)
    
    # samples, fmaps, pck plot
    torch.save(x_sampled, f"{save_folder}/sampled_{dataset_name}_{subset_fraction}.pt")
    torch.save(fmap_sampled, f"{save_folder}/fmap_sampled_{dataset_name}_{subset_fraction}.pt")
    plt.savefig(f"{save_folder}/pck_{dataset_name}_{subset_fraction}.png")
    
    
    