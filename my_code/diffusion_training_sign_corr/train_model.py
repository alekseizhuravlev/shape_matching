import torch
from tqdm import tqdm


def train_epoch(model, is_unconditional,
                train_dataloader, noise_scheduler,
                opt, loss_fn):
    
    # Keeping a record of the losses for later viewing
    losses = []

    # The training loop
    for x, y in tqdm(train_dataloader, total=len(train_dataloader)):

        # Unpack the batch
        x = x.to(model.device()) 
        y = y.to(model.device())
        
        # sample the noise and the timesteps
        noise = torch.randn_like(x)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (x.shape[0],)).long().to(model.device())
        
        # Add the noise to the input
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Get the model prediction
        if is_unconditional:
            pred = model(sample=noisy_x, timestep=timesteps).sample
        else:
            pred = model(sample=noisy_x, timestep=timesteps, conditioning=y).sample

        # Calculate the loss
        loss = loss_fn(pred, noise) # How close is the output to the noise

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        losses.append(loss.item())

    return model, losses