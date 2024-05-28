import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def count_zero_regions(x_sampled, threshold, percentage):
    incorrect_zero_indices = []
    
    for i in range(x_sampled.shape[0]):
        if (x_sampled[i] > threshold).int().sum() > percentage * x_sampled[i].numel():
            incorrect_zero_indices.append(i)
            
    print(f'Incorrect zero regions: {len(incorrect_zero_indices)} / {x_sampled.shape[0]} = '
          f'{len(incorrect_zero_indices) / x_sampled.shape[0]*100:.2f}%')
    
    return incorrect_zero_indices