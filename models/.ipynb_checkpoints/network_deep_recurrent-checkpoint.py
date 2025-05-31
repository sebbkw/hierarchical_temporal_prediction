import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize

from models.network_base import NetworkBase

class NetworkDeepRecurrent (NetworkBase):
    # Hidden units is an array of ints corresponding to
    # number of hidden units in each group
    def __init__ (self, hyperparameters):
        super(NetworkDeepRecurrent, self).__init__()

        self.loss                  = hyperparameters['loss']
        self.lam                   = hyperparameters['lam']
        self.frame_size            = hyperparameters['frame_size']
        self.warmup                = hyperparameters['warmup']
        self.device                = hyperparameters['device']

        self.hidden_units_groups     = [int(i) for i in hyperparameters['hidden_units_groups'].split(',') if len(i)]
        self.hidden_units_groups_len = len(self.hidden_units_groups)
        self.hidden_units            = self.hidden_units_groups[0]

        self.output_units_groups     = [self.frame_size]
        self.output_units_groups_len = len(self.output_units_groups)
        self.output_units            = sum(self.output_units_groups)

        self.rnn = nn.RNN(
            input_size = self.frame_size,
            hidden_size = self.hidden_units,
            num_layers = self.hidden_units_groups_len,
            nonlinearity = 'relu',
            batch_first = True
        )
        self.fc = nn.Linear(
            in_features = self.hidden_units,
            out_features = self.output_units
        )

        with torch.no_grad():
            self.state_dict()['rnn.weight_hh_l0'][:] = nn.Parameter(torch.eye(self.hidden_units, self.hidden_units)) / 100

    def preprocess_data (self, data):
        x, y = data

        noise = torch.randn(x.shape)*0.5011
        x += noise
        y += noise

        x = x.to(self.device)#[:, :, :20, :20].reshape(-1, 50, 20*20)
        y = y.to(self.device)#[:, :, :20, :20].reshape(-1, 50, 20*20)

        return x, y

    def forward (self, inputs):
        rnn_outputs, _ = self.rnn(inputs)
        fc_outputs = self.fc(rnn_outputs)

        return fc_outputs, rnn_outputs
