import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def predict_sign_change(net, verts, faces, evecs_flip, evecs_cond):
    
    # normalize the evecs
    evecs_flip = torch.nn.functional.normalize(evecs_flip, p=2, dim=1)
    
    if evecs_cond is not None:
        evecs_cond = torch.nn.functional.normalize(evecs_cond, p=2, dim=1)
        evecs_input = torch.cat([evecs_flip, evecs_cond], dim=-1)
    else:
        evecs_input = evecs_flip
        
    # process the flipped evecs
    support_vector_flip = net(
        verts=verts,
        faces=faces,
        feats=evecs_input,
    ) # [1 x 6890 x 1]

    # normalize the support vector
    support_vector_norm = torch.nn.functional.normalize(support_vector_flip, p=2, dim=1)
    
    # multiply the support vector by the flipped evecs [1 x 6890 x 4].T @ [1 x 6890 x 4]
    product_with_support = support_vector_norm.transpose(1, 2) @ evecs_flip

    if product_with_support.shape[1] == product_with_support.shape[2]:
        # take only diagonal elements
        sign_flip_predicted = torch.diagonal(product_with_support, dim1=1, dim2=2)
        
    # get the sign of the support vector
    # sign_flip_predicted = product_with_support
 
    return sign_flip_predicted, support_vector_norm, product_with_support


