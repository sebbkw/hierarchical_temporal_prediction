import sys

from VirtualNetworkPhysiology import VirtualPhysiology

sys.path.append("../")
from models.network_hierarchical_recurrent import NetworkHierarchicalRecurrent


# Script wide variables
DEVICE     = 'cpu'

MODEL_PATH = '/home/seb/rnn_hierarchical/model_checkpoints/20x20_3x800_noDale_-5.75L1_Oct14-21-05/2000-epochs_model.pt'
SAVE_PATH = '/media/seb/Elements/rnn_emil/rnn_hierarchical/20x20_3x800_-575_noDale.pickle'

# Load network checkpoint
model, hyperparameters, _ = NetworkHierarchicalRecurrent.load(
    model_path=MODEL_PATH, device=DEVICE
)

# Instantiate new VirtualPhysiology object
vphys = VirtualPhysiology(
    model=model,
    hyperparameters=hyperparameters,
    frame_shape=(20, 20),
    hidden_units=[800, 800, 800],
    device=DEVICE
)


# Run virtual physiology methods
vphys.get_response_weighted_average(n_rand_stimuli=20000) \
     .get_grating_responses() \
     .get_grating_responses_parameters() \
     .save(SAVE_PATH)
