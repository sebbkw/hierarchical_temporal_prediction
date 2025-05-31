import losses.offset_mse as l

def temporal_prediction (self, out, data):
    return l.offset_mse(self, 1, out, data)
