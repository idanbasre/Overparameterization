import torch
import torch.nn as nn


class linear_n_dpeth(nn.Module):
  '''
  Linear Netwrok
  '''
  def __init__(self, layers_width):
    super().__init__()

    self.layers = nn.ModuleList([])
    for (i, layer_width) in enumerate(layers_width[0:-1]):
      next_layer_width = layers_width[i+1]
      self.layers.append(nn.Linear(layer_width, next_layer_width, bias=False))
      torch.nn.init.normal_(self.layers[i].weight, mean=0.0, std=0.01)

  def forward(self, x):
    output = x
    for layer in self.layers:
      output = layer(output)

    return output