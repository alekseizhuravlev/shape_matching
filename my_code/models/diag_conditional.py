import torch
from torch import nn
from torch.nn import functional as F

from diffusers import UNet2DModel


class DiagConditionedUnet(nn.Module):
  def __init__(self, params_dict):
    super().__init__()

    self.model = UNet2DModel(**params_dict)


  def forward(self, sample, timestep, conditioning):

    # Create a diagonal matrix from the conditioning
    conditioning_diag = torch.diag_embed(conditioning) 

    # concatenate the sample and the conditioning
    net_input = torch.cat((sample, conditioning_diag), 1) # (bs, 2, 28, 28)

    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, timestep)

  def device(self):
    return self.model.device
  

  