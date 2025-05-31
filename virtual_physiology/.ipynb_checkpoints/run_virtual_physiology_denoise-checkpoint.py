import sys
sys.path.append("../")
#from models.network_recurrent import NetworkRecurrent as Network
from models.network_recurrent_masked import NetworkRecurrentMasked as Network
from VirtualNetworkPhysiology import VirtualPhysiology

# Script wide variablesscr
DEVICE     = 'cpu'

MODEL_PATH = '/media/seb/Elements/rnn_models/rnn_refactor6/denoise_-7.0L1_Aug11-12-38/2000-epochs_model.pt'
SAVE_PATH  = '/media/seb/Elements/rnn_emil/alternative_models/denoise_traintestsplit_7_2000.pickle'

# Load network checkpoint
model, hyperparameters, _ = Network.load(
    model_path=MODEL_PATH, device=DEVICE
)

# Instantiate new VirtualPhysiology object
vphys = VirtualPhysiology(
    model=model,
    hyperparameters=hyperparameters,
    frame_shape=(36, 36),
    hidden_units=[2592],
    device=DEVICE
)


# Run virtual physiology methods
vphys.get_response_weighted_average(n_rand_stimuli=20000) \
     .get_grating_responses() \
     .get_grating_responses_parameters() \
     .save(SAVE_PATH)
