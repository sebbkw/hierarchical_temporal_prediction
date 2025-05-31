################################
# Imports and helper functions #
################################

import argparse, sys
import torch

import train

##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Path to model checkpoint')
is_arg_required = not '--path' in sys.argv
parser.add_argument('--name', required=is_arg_required)
parser.add_argument('--L1', type=float, required=is_arg_required)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##############################
# Optionally load checkpoint #
##############################

if not args.path is None:
    train.main({'device': device}, args.path)
    exit()

##################################
# Define various hyperparameters #
##################################

hyperparameters = {
    "device"   : device,
    "save_dir" : 'model_checkpoints',
    "name"     : args.name,

    "batch_size"  : 100,
    "epochs"      : 2000,
    "lr"          : 10**-4,
    "checkpoints" : 1000,

    "dataset" : '', # dataset name
    "loss"    : '', # loss name
    "model"   : '', # model_name

    "frame_size"            : 20*20,
    "hidden_units_groups"   : '800,800,800',
    "beta_weights"          : '0.9,0.1,0.1',
    "lam"                   : 10**args.L1,
    "warmup"                : 4,
    "local_inhibitory_prop" : 0.2
}


print('Using hyperparameters:')
for k, v in hyperparameters.items():
    print('\t', k, '\t', v)
print('')

train.main(hyperparameters)
