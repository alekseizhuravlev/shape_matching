import torch
from tqdm import tqdm

def train_epoch(model, is_unconditional,
                train_dataloader, noise_scheduler,
                opt, loss_fn, lr_scheduler, accelerator=None):
    
    # Keeping a record of the losses for later viewing
    losses = []

    # The training loop
    for x, y in tqdm(train_dataloader, total=len(train_dataloader), disable=accelerator is None or not accelerator.is_local_main_process):

        # Unpack the batch
        
        if accelerator is None:
            x = x.to(model.device()) 
            y = y.to(model.device())
        
        # sample the noise and the timesteps
        noise = torch.randn_like(x)
        
        if accelerator is None:
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (x.shape[0],)).long().to(model.device())
        else:
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (x.shape[0],)).long().to(accelerator.device)
        
        # Add the noise to the input
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Get the model prediction
        if is_unconditional:
            pred = model(sample=noisy_x, timestep=timesteps).sample
        else:
            pred = model(sample=noisy_x, timestep=timesteps, conditioning=y).sample

        # Calculate the loss
        loss = loss_fn(pred, noise) # How close is the output to the noise
        
        # if accelerator is None:
        #     loss = loss_fn(pred, noise) # How close is the output to the noise
        # else:
        #     with accelerator.autocast():
        #         loss = loss_fn(pred, noise)

        # Backprop and update the params:
        opt.zero_grad()
        
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
            
        opt.step()
        lr_scheduler.step()

        # Store the loss for later
        losses.append(loss.item())

    return model, losses