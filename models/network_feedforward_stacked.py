import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize

from models.network_base import NetworkBase
from models.network_feedforward import NetworkFeedforward

class NetworkFeedforwardStacked (NetworkBase):
    def __init__ (self, paths):
        super(NetworkFeedforwardStacked, self).__init__()

        self.stacks = []
        
        for path in paths:
            print(path)
            lower_stack, _, _ = NetworkFeedforward.load(path, device='cpu')
            lower_stack = lower_stack.eval()
            
            self.stacks.append(lower_stack)

    def preprocess_data (self, data):
        x, y = data
        
        noise = torch.randn(x.shape)*0.5011
        x += noise
        y += noise

        x = x.to(self.stacks[0].device)
        y = y.to(self.stacks[0].device)

        return x, y
            
    def forward (self, inputs):
        inputs = inputs.unsqueeze(1)[:, :, :]
        
        stack_hidden_states = []
        
        for stack_idx, stack in enumerate(self.stacks):
            if stack_idx == 0:
                hidden_state, future_frames = stack(inputs)
            else:
                hidden_state = stack(stack_hidden_states[-1])[0]
                
            stack_hidden_states.append(hidden_state)
        
        final_hidden_states = []
        for h in stack_hidden_states:
            p = inputs.shape[2]-h.shape[2]
            h = h[:, :, :, 0]
            s = list(h.shape)
            s[2] = p
            h = torch.cat([torch.zeros(s), h], dim=2)
            final_hidden_states.append(h)
        final_hidden_states = torch.cat(final_hidden_states, dim=1)
        final_hidden_states = final_hidden_states.transpose(2, 1)
                
        return future_frames, final_hidden_states