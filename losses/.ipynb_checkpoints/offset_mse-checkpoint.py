import torch.nn as nn

def offset_mse (self, temporal_offset, out, data):
    predictions, hidden = out
    _, frame_targets    = data

    if temporal_offset > 0:
        predictions = predictions[:, self.warmup:-temporal_offset]
    else:
        predictions = predictions[:, self.warmup:]

    frame_targets = frame_targets[:, self.warmup+temporal_offset:, :]

    MSE = nn.functional.mse_loss(predictions, frame_targets)

    ret = { 'L1': self.L1(), 'mse0': MSE }
    loss = MSE + ret['L1']

    return loss, ret
