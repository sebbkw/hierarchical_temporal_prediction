#### Import modules ###

import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append('../')

from models.network_hierarchical_recurrent import NetworkHierarchicalRecurrent
from models.network_feedforward_stacked import NetworkFeedforwardStacked

DEVICE = 'cpu'

def get_layer_idxs (layer, units_per_layer=800):
    return slice(units_per_layer*layer, units_per_layer*(layer+1))

def pad_matrix (m, pad_size):
    # repeat along last (feature) dimension
    m = torch.repeat_interleave(m.unsqueeze(0), pad_size, dim=0)

    # offseting by one for each repeat along the last dimensions
    m_stacked = []
    for i in range(pad_size):
        start_idx = i
        end_idx   = -(pad_size-i-1)

        if i != pad_size-1:
            m_stacked.append(m[start_idx, :, start_idx:end_idx])
        else:
            m_stacked.append(m[start_idx, :, start_idx:])

    # stack along last (feature) dimension
    m_stacked = torch.stack(m_stacked, dim=2)

    # now flatten last two dimensions
    m_stacked = torch.flatten(m_stacked, start_dim=2)

    return m_stacked

# Get model responses
def get_rnn_responses(stimuli_processed, model, layer):
    stimuli_reshaped = stimuli_processed.reshape(1, -1, 20*40)

    with torch.no_grad():
        _, hidden_states = model(torch.Tensor(stimuli_reshaped))
        h = hidden_states.numpy()[0, :, get_layer_idxs(layer)]
    rnn_model_responses = h

    return rnn_model_responses


# Load raw natural movie stimuli
mov_stimuli = np.load(
    '', # dir to the stimuli
    allow_pickle=True
).item()

paths = [
    # paths of the model checkpoints
]

names = [
    # name to use for each model checkpoint
]


for name, path in zip(names, paths):
    print('Starting', path)

    full_path = f'/home/seb/rnn_hierarchical/model_checkpoints/{path}'
    model, hyperparameters, _ = NetworkHierarchicalRecurrent.load(model_path=path, device=DEVICE)

    for layer in range(3):
        full_name = f'{name}_layer{layer}'

        model_outputs = {}

        for stim, stim_data in mov_stimuli.items():
            stim_responses = get_rnn_responses(stim_data, model, layer)

            print(stim_responses.shape)
            model_outputs[stim] = stim_responses

        save_path = f'../neural_fitting/model_data/{full_name}.npy'
        np.save(save_path, model_outputs)
        print(save_path)
