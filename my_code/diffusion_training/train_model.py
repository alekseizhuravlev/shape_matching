import torch
from tqdm.auto import tqdm


def train(model, n_epochs, loss_fn, is_unconditional, train_dataloader, noise_scheduler):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # The optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Keeping a record of the losses for later viewing
    losses = []

    iterator = tqdm(range(n_epochs))
    # The training loop
    for epoch in iterator:
        for x, y in train_dataloader:

            # Get some data and prepare the corrupted version
            x = x.to(device) #* 2 - 1 # Data on the GPU (mapped to (-1, 1))
            y = y.to(device)
            
            noise = torch.randn_like(x)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (x.shape[0],)).long().to(device)
            
            
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            # Get the model prediction
            if is_unconditional:
                pred = model(sample=noisy_x, timestep=timesteps).sample
            else:
                # For the conditional model, we need to pass in the class labels as well
                pred = model(sample=noisy_x, timestep=timesteps, conditioning=y).sample

            # Calculate the loss
            loss = loss_fn(pred, noise) # How close is the output to the noise

            # Backprop and update the params:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Store the loss for later
            losses.append(loss.item())

        # Print out the average of the last 100 loss values to get an idea of progress:
        avg_loss = sum(losses[-100:])/100
        # print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')
        iterator.set_postfix({'avg_loss': avg_loss})

    return model, losses