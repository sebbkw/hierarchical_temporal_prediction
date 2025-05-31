import torch.nn as nn

def hierarchical_temporal_prediction_feedforward (self, out, data):
    PADDING         = self.padding - 1
    TEMPORAL_OFFSET = 1
    
    prediction, hidden = out
    _, frame_targets    = data
    
    ##############################
    # Predictions for each group #
    ##############################

    predictions_by_group = []
    for p in prediction:
        if TEMPORAL_OFFSET > 0:
            predictions_by_group.append(p[:, :-TEMPORAL_OFFSET])
        else:
            predictions_by_group.append(p)

    ##########################
    # Targets for each group #
    ##########################

    targets_by_group = [frame_targets[:, PADDING+TEMPORAL_OFFSET:, :]]
    for h in hidden[:-1]:
        targets_by_group.append(h[:, PADDING+TEMPORAL_OFFSET:])

    #######################
    # MSEs for each group #
    #######################

    loss = 0
    ret  = {}
    
    MSEs = []
    for group_idx, (p, t) in enumerate(zip(predictions_by_group, targets_by_group)):
        MSE = nn.functional.mse_loss(p, t)
        ret[f'mse{group_idx}'] = MSE
        loss += MSE
        
    L1   = self.L1()
    loss += L1
    ret['L1'] = L1

    return loss, ret
