import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize

from models.network_base import NetworkBase

class NetworkHierarchicalFeedforward (NetworkBase):
    def __init__ (self, hyperparameters):
        super(NetworkHierarchicalFeedforward, self).__init__()

        self.loss                  = hyperparameters['loss']
        self.lam                   = hyperparameters['lam']
        self.frame_size            = hyperparameters['frame_size']
        self.warmup                = hyperparameters['warmup']
        self.device                = hyperparameters['device']

        self.hidden_units_groups     = [int(i) for i in hyperparameters['hidden_units_groups'].split(',') if len(i)]
        self.hidden_units_groups_len = len(self.hidden_units_groups)
        self.hidden_units            = sum(self.hidden_units_groups)
        self.beta_weights            = [1 for _ in range(len(self.hidden_units_groups))]

        self.output_units_groups     = [self.frame_size] + self.hidden_units_groups[:-1]
        self.output_units_groups_len = len(self.output_units_groups)
        self.output_units            = sum(self.output_units_groups)

        self.padding = 5
        
        for stack_i in range(len(self.hidden_units_groups)):
            ih_in_channels  = 1 if stack_i==0 else self.hidden_units_groups[stack_i-1]
            ih_out_channels = self.hidden_units_groups[stack_i]
            ih_kernel       = (
                self.padding,
                self.frame_size if stack_i==0 else 1
            )
            setattr(self, f'ih{stack_i}', nn.Conv2d(
                in_channels=ih_in_channels,
                out_channels=ih_out_channels,
                kernel_size=ih_kernel,
                #padding=((self.padding-1)//2, 1)
            ))
            
            fc_in_features  = self.hidden_units_groups[stack_i]
            fc_out_features = self.output_units_groups[stack_i]
            setattr(self, f'fc{stack_i}', nn.Linear(
                in_features=fc_in_features,
                out_features=fc_out_features
            ))

        self.ReLU = nn.ReLU()
        
        self.initialize_weights_biases()

    def initialize_weights_biases (self):
        for stack_i in range(len(self.hidden_units_groups)):
            ih_layer = getattr(self, f'ih{stack_i}')
            fc_layer = getattr(self, f'fc{stack_i}')

            nn.init.uniform_(ih_layer.bias, 0, 0)
            nn.init.uniform_(fc_layer.bias, 0, 0)

            I = self.frame_size if stack_i==0 else self.hidden_units_groups[stack_i-1]
            J = self.hidden_units_groups[stack_i]
            S = 100

            ih_upper_bound = np.sqrt(2/(I*S))
            fc_upper_bound = np.sqrt(2/(I*J))
            nn.init.uniform_(ih_layer.weight, 0, ih_upper_bound)
            nn.init.uniform_(fc_layer.weight, 0, fc_upper_bound)
        
    def preprocess_data (self, data):
        x, y = data

        noise = torch.randn(x.shape)*0.5011
        x += noise
        y += noise

        x = x.to(self.device)
        y = y.to(self.device)

        return x, y
    
    def forward (self, inputs):        
        hidden = []
        output = []
        
        for stack_i in range(len(self.hidden_units_groups)):
            ih_layer = getattr(self, f'ih{stack_i}')
            fc_layer = getattr(self, f'fc{stack_i}')
            
            if stack_i == 0:
                i = inputs.unsqueeze(1)
            else:
                i = torch.transpose(hidden[-1], 1, 2).unsqueeze(3)
                i = i.detach().clone()
                                
            h = self.ReLU(ih_layer(i))
            h = torch.transpose(h[:, :, :, 0], 1, 2)            
            o = fc_layer(h)
            
            hidden.append(h)
            output.append(o)

        # hidden_state = torch.cat(hidden, dim=2)
        # predictions  = torch.cat(output, dim=2)
        
        return output, hidden