import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize

from models.network_base import NetworkBase

class NetworkFeedforward (NetworkBase):
    def __init__ (self, hyperparameters):
        super(NetworkFeedforward, self).__init__()

        self.hyperparameters = hyperparameters
        
        self.stack       = hyperparameters['stack']
        #self.lower_stack = lower_stack
        self.units       = hyperparameters['units']
        self.lam         = hyperparameters['lam']
        self.device      = hyperparameters['device']

        self.frame_size         = 800
        self.hidden_state_size  = self.frame_size if self.stack == 0 else 1

        self.in_conv_channels  = 1 if self.stack == 0 else self.units
        self.out_conv_channels = self.units
        self.kernel_size_t     = hyperparameters['kernel_size_t']
        self.kernel            = (self.kernel_size_t, self.hidden_state_size)
        self.kernel_T          = (1, self.hidden_state_size)

        self.ReLU   = nn.ReLU()
        self.conv   = nn.Conv2d(self.in_conv_channels, self.out_conv_channels, self.kernel)
        self.conv_T = nn.ConvTranspose2d(self.out_conv_channels, self.in_conv_channels, self.kernel_T)

    def get_inputs (self, inputs):
        if True:
            return inputs
        else:
            print(inputs.shape, self.lower_stack(inputs)[0].shape)
            return self.lower_stack(inputs)[0]

    def get_targets (self, targets):
        t = targets[:, :, self.kernel_size_t:]
        if True:
            return t
        else:
            print('lower targets')
            return self.lower_stack(t)[0]

    def forward (self, x):
        inputs      = self.get_inputs(x)        
        hidden      = self.ReLU(self.conv(inputs))
        predictions = self.conv_T(hidden)
        
        return hidden, predictions

    def loss_fn (self, frame_predictions, frame_targets):
        targets = self.get_targets(frame_targets)

        mse = nn.functional.mse_loss(
            targets,
            frame_predictions[:, :, :-1]
        )

        weights = torch.empty(0, device=self.device)
        for name, params in self.named_parameters():
            if 'weight' in name:
                weights = torch.cat((weights, params.flatten()), 0)
                
        L1 = self.lam*weights.abs().sum()

        return mse+L1, {'mse': mse, 'L1': L1}