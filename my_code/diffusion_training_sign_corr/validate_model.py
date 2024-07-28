import torch

import my_code.diffusion_training_sign_corr.sample_model as sample_model
import my_code.diffusion_training_sign_corr.evaluate_samples as evaluate_samples


def validate_epoch(model, noise_scheduler, test_dataset, sign_corr_net):
    
    model.eval()
    
    # unpack the validation payload
    # test_dataloader = val_payload['dataloader']
                
    # sample the model
    # x_sampled = sample_model.sample(model, test_dataloader, noise_scheduler)  
    
    with torch.no_grad():
        fmap_sampled = sample_model.sample_dataset(model, test_dataset, noise_scheduler)

        ### assign gt signs and unnormalize the samples 
        # x_gt = torch.stack([test_dataset[i]['second']['C_gt_xy'] for i in range(len(test_dataset))])
        # fmap_sampled = torch.sign(x_gt) * (x_sampled + 1) / 2
        
        
        ### calculate metrics and pck
        metrics_payload, fig_pck = evaluate_samples.calculate_metrics(
            fmap_sampled,
            test_dataset,
            sign_corr_net
        )
    # calculate mean, median, min, max...
    # metrics_payload = evaluate_samples.preprocess_metrics(metrics)
    # plot pck
    # fig_pck = evaluate_samples.plot_pck(metrics, title=f"PCK")
    
    model.train()
    
    return model, metrics_payload, {'pck': fig_pck}