import losses.hierarchical_offset_mse as l

def hierarchical_temporal_prediction_t5 (self, out, data):
    return l.hierarchical_offset_mse(self, 5, out, data)
