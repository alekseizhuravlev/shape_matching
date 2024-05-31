import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def sample(model, n_samples, noise_shape, conditioning, noise_scheduler, plot_last_steps):

    # device = model.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare random x to start from, plus some desired labels y
    x = torch.randn(n_samples, 1, noise_shape, noise_shape).to(device)      
        
    # Sampling loop
    for i, t in tqdm(enumerate(noise_scheduler.timesteps), total=noise_scheduler.config.num_train_timesteps):

        # Get model pred
        with torch.no_grad():
            if conditioning is None:
                residual = model(x, t).sample
            else:
                residual = model(x, t,
                                 conditioning=conditioning.to(device)
                                 ).sample

        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample
        
        if plot_last_steps and i > 900 and i % 10 == 0:
            plt.imshow(x[0][0].cpu().numpy())
            plt.show()
        
    return x