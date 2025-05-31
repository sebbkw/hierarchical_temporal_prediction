import sys
from VirtualNetworkPhysiology import VirtualPhysiology

sys.path.append("../")
from models.network_hierarchical_recurrent import NetworkHierarchicalRecurrent

DEVICE     = 'cpu'
MODEL_PATH = '' # path to the saved model checkpoint
VPHYS_PATH = '' # path to the saved virtual physiology object                             

# Load network checkpoint
model, hyperparameters, _ = NetworkHierarchicalRecurrent.load(
    model_path=MODEL_PATH, device=DEVICE
)

# Instantiate new VirtualPhysiology object
vphys = VirtualPhysiology.load(
    data_path=VPHYS_PATH,
    model=model,
    hyperparameters=hyperparameters,
    frame_shape=(20, 20),
    hidden_units=[800, 800, 800],
    device=DEVICE
)

vphys.get_plaid_pattern_index()
vphys.save(VPHYS_PATH)
