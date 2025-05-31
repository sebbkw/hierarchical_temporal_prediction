######################################################
# Loads stimuli, passes them through VGG network and #
# stores the output for a given layer                #
######################################################

import os
import h5py
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

BASE_DIR = './'

#################################################
# Saves dataset by either creating or appending #
# to .hdf5 file at specified path               #
#################################################

def save_data_hdf5(data, path_name, datasdet_name):
    # Create the file or just read/write if it already exists
    path_name = os.path.join(BASE_DIR, path_name)

    f = h5py.File(path_name, 'a')

    # Check if the dataset already exists
    if dataset_name in f:
        dset = f[dataset_name]
        old_len = len(dset)

        # And resize it to append more data
        dset.resize(len(data)+len(dset), axis=0)
        dset[old_len:] = data

        return len(dset)
    elif len(data):
        # If not, create it
        maxshape = (None, *data.shape[1:])
        dset = f.create_dataset(dataset_name, data=data, maxshape=maxshape, chunks=True)
        return len(dset)

    f.close()


##################################
# First load and process stimuli #
##################################

# Load raw natural movie stimuli
mov_stimuli = np.load(
    '', # dir for natural stimuli
    allow_pickle=True
).item()


##################################
# Pass each clip through VGG     #r
##################################

# Load pre-trained vgg net
vgg19 = torchvision.models.vgg19(pretrained=True)

# Transforms to convert images into format required for vgg network
preprocess_frame = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_vgg_featuremaps (clip, layer):    
    # Images must be 3 channels (rgb), so stack grayscale image
    clip = np.concatenate([
        np.zeros((clip.shape[0], 10, 40)),
        clip,
        np.zeros((clip.shape[0], 10, 40))        
    ], axis=1)
    
    clip_stacked = torch.Tensor(np.stack([clip, clip, clip], axis=1))

    clip_transformed = torch.empty((clip_stacked.shape[0], 3, 40, 40))

    for frame_idx, frame in enumerate(clip_stacked):
        # Apply transforms
        frame_normed = preprocess_frame(frame)
        clip_transformed[frame_idx] = frame_normed
        
    # Pass through network
    print('Starting', vgg19.features[:layer][-1])
    feature_map = vgg19.features[:layer](clip_transformed)
    
    feature_map = feature_map.reshape(feature_map.shape[0], -1).cpu().detach().numpy()
    
    return feature_map

    

# Loop through each convolutional layer 1-16 (after ReLU)
conv_layer_idxs = [1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29, 31, 33, 35]

for layer_idx, layer_n in enumerate(conv_layer_idxs):
    layer_name = 'conv{}'.format(layer_n)

    print('Starting convolutional layer', layer_n)

    model_outputs = {}

    # Loop through all clips and images to get feature map
    for stim, stim_data in mov_stimuli.items():
        print('\tProcessing stimuli', stim)

        # Get featuremap then pass through batchnorm
        feature_map = get_vgg_featuremaps(stim_data.reshape(-1, 20, 40), layer=layer_n+1)
        print('\t', feature_map.shape)

        chunk_size = 5
        chunked_clip = []
        for i in range(0, feature_map.shape[0], chunk_size):
            clip = feature_map[i:i+chunk_size].mean(axis=0)
            chunked_clip.append(clip)
        chunked_clip = np.array(chunked_clip)

        print('\t', chunked_clip.shape)

        model_outputs[stim] = chunked_clip

    np.save(f'./model_data/vgg19_fixed_conv{layer_idx}.npy', model_outputs)

print('Processed all data')
