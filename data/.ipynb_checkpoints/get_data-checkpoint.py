import os, pickle, glob, math
import numpy as np
import cv2
import h5py
import torch
import scipy.ndimage as ndimage
import skimage.filters as filters

import matplotlib.pyplot as plt


# Taken from https://github.com/yossing/temporal_prediction_model
def whiten_and_filter_image(im_to_filt):
    N = im_to_filt.shape[0]
    imf=np.fft.fftshift(np.fft.fft2(im_to_filt))
    f=np.arange(-N/2,N/2)
    [fx, fy] = np.meshgrid(f,f)
    [rho,theta]=cart2pol(fx,fy)
    filtf = rho*np.exp(-0.5*(rho/(0.7*N/2))**2)
    imwf = filtf*imf
    imw = np.real(np.fft.ifft2(np.fft.fftshift(imwf)))
    return imw
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def crop_square (im):
    # Do square crop (crop along longest side)
    height, width = im.shape[:2]
    shortest_side = min(height, width)
    square_crop_height = int( (height-shortest_side)/2 )
    square_crop_width = int( (width-shortest_side)/2 )
    im = im[square_crop_height:height-square_crop_height, square_crop_width:width-square_crop_width]
    return im

def get_array_from_video (path_name, n_frames):
    # Open video file and check if it can be read
    cap = cv2.VideoCapture(path_name)
    if (cap.isOpened()== False): 
        raise Exception("Error opening file '{}'".format(path_name))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    shortest_side = min(frame_width, frame_height)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 118:
        n_clips = int(total_frames/4/n_frames)
    elif fps > 48 and fps < 62:
        n_clips = int(total_frames/2/n_frames)
    elif fps < 32 and fps > 23:
        n_clips = int(total_frames/n_frames)
    else:
        raise Exception("FPS below 23 ({})".format(fps))

    resize = 600
    if frame_width < resize:
        resized_width = frame_width
    else:
        resized_width = resize

    resized_height = int(frame_height/frame_width*resized_width)
    #resized_width = resized_height = min(resized_width, resized_height)

    # Pre-allocate array for speed
    clips_arr = np.zeros((
        n_clips,
        resized_height,
        resized_width,
        n_frames
    ))

    print('\t... Frame size {} x {} pixels'.format(frame_width, frame_height))

    if (min(frame_width, frame_height) < resize): 
        raise Exception("Minimum dimension must be {} pixels for file '{}'".format(resize, path_name))

    # Read video frame-by-frame and split into n_frames length clips
    frame_idx_30fps = 0

    for frame_idx in range(total_frames):
        if (fps > 118) and (frame_idx%4 == 1):
            continue
        elif (fps > 48 and fps < 62) and (frame_idx%2 == 1):
            continue

        # Get index of current clip and current frame in that clip
        curr_clip = frame_idx_30fps // n_frames
        curr_frame = frame_idx_30fps % n_frames

        # Curr_clip should always be one less than n_clips
        # as any overflow will mean final clip is not completely filled
        if curr_clip == n_clips:
            break

        ret, frame = cap.read()

        if ret and (frame_idx%4 == 0):
            if frame_idx % 1000 == 0:
                print('\t... Processing frame {}/{}'.format(frame_idx, total_frames))

            # Convert to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resise using bilinear interpolation (Singer, 2018)
            frame = cv2.resize(
                frame,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LINEAR
            )

            # Crop
            #frame = crop_square(frame)

            # Filter
            #frame = whiten_and_filter_image(frame)
            frame = filters.difference_of_gaussians(frame, 1)

            clips_arr[curr_clip, :, :, curr_frame] = frame
            frame_idx_30fps += 1

        elif ret == False:
            raise Exception("Error loading frame {}, aborting file".format(frame_idx))

    cap.release()

    return clips_arr

# Crops frames in each clip to given crop_size
# Returns array in form (clips, crop_size_row, crop_size_col, n_frames)
def crop_clip (frames_arr, crop_size, x_crops, y_crops, padding):
    n_frames = frames_arr.shape[2]
    y_pixels, x_pixels = frames_arr.shape[:2]
    crop_size_y, crop_size_x = crop_size

    # Only start cropping inside 'padded' area
    start_x = int(x_pixels*padding)
    start_y = int(y_pixels*padding)

    total_x = int(x_pixels*(1 - padding*2))
    total_y = int(y_pixels*(1 - padding*2))

    # Pre-allocate cropped frames array
    cropped_frames_arr = np.zeros((
        x_crops*y_crops,
        crop_size_y,
        crop_size_x,
        n_frames
    ))

    # Work out step between crops
    # given number of crops, crop size and padding
    overlap_x = ((crop_size_x*x_crops)-total_x) / max(x_crops-1, 1)
    step_x = crop_size_x-overlap_x
    overlap_y = ((crop_size_y*y_crops)-total_y) / max(y_crops-1, 1)
    step_y = crop_size_y-overlap_y

    x_range = np.arange(0, x_crops) * step_x
    y_range = np.arange(0, y_crops) * step_y

    # Iterate across each frame in x_range and y_range
    i = 0
    for x_pos in x_range:
        for y_pos in y_range:
            x = int(start_x + x_pos)
            y = int(start_y + y_pos)

            cropped_frames_arr[i] = frames_arr[y:y+crop_size_y, x:x+crop_size_x]
            i += 1

    return cropped_frames_arr

# Returns array in form (n_clips, crop_size_row, crop_size_col, n_frames)
def process_data (clips_arr, crop_size, x_crops, y_crops, padding):
    n_clips = clips_arr.shape[0]
    n_frames = clips_arr.shape[3]
    total_crops = x_crops*y_crops

    # Pre-allocate array for the set of
    # cropped clips for each uncropped clip
    cropped_clips_arr = np.zeros((
        n_clips*total_crops,
        crop_size[0],
        crop_size[1],
        n_frames
    ))

    # Loop through each uncropped examples
    # to fill pre-allocated array with set of cropped frames
    for curr_clip_idx, clip in enumerate(clips_arr):
        cropped_clip_slice = slice(curr_clip_idx*total_crops, (curr_clip_idx+1)*total_crops)
        cropped_clips_arr[cropped_clip_slice] = crop_clip(clip, crop_size, x_crops, y_crops, padding)

        # Free up memory
        # clips_arr = clips_arr[1:]

    return cropped_clips_arr

# Returns array in form (n_examples, frames, crop_size**2)
def reshape_data (data):
    data_shape = data.shape
    reshaped_shape = (data_shape[0], data_shape[1], data_shape[2]*data_shape[3])
    data = np.reshape(data, reshaped_shape)

    return data

# Normalized each individual clip
# Returns array in form (n_examples, frames, crop_size**2)
def normalize_data (data):
    examples = []

    for i, clip in enumerate(data):
        if np.std(clip) > 0:
            examples.append( (clip - np.mean(clip)) / np.std(clip) )
        else:
            print('\t... Clip', i, ' has 0 std (reject)')

    return np.array(examples)

def reject_low_motion_clips (data):
    # Get a subset of clips for speed
    random_idxs = np.random.choice(data.shape[0], min(2000, data.shape[0]))
    random_data = data[random_idxs]

    # Construct normally distributed dataset
    mn, std = np.mean(random_data), np.mean(np.std(random_data, axis=1))
    noise_shape = (random_data.shape[0], random_data.shape[1], 10*10)
    noise_dataset = np.random.normal(mn, std, size=noise_shape)
    print(noise_dataset.shape)
    # Get MSE from data and construct threshold based on 95th percentile
    mse = np.median(np.mean(np.diff(noise_dataset, axis=1)**2, axis=1), axis=1)
    thresh = np.percentile(mse, 25)

    # Split clip into small sections
    def split_clip (clip, thresh):
        clip = clip.reshape(-1, 36, 36)

        x_step = 10
        y_step = 10

        x_range = np.arange(0, 6) * x_step
        y_range = np.arange(0, 3) * y_step

        thresh_count = 0

        # Iterate across each frame in x_range and y_range
        for x_pos in x_range:
            for y_pos in y_range:
                split = clip[:, y_pos:y_pos+y_step, x_pos:x_pos+x_step].reshape(clip.shape[0], -1)
                split_mse = np.mean(np.diff(split, axis=0)**2)

                if split_mse > thresh:
                    thresh_count += 1

        return thresh_count

    # Iterate through each clip in the dataset
    filtered_clips = []
    for clip_idx, clip in enumerate(data):
        clip_mse_count = split_clip(clip, thresh)

        if clip_mse_count > 0:
            filtered_clips.append(clip)

    return np.array(filtered_clips)

# Saves dataset by either creating or appending to .hdf5 file at specified path
def save_data_hdf5(data, path_name):
    # Create the file or just read/write if it already exists
    f = h5py.File(path_name, 'a')

    # Check if the dataset already exists
    if "clips" in f:
        dset = f["clips"]
        old_len = len(dset)

        # And resize it to append more data
        dset.resize(len(data)+len(dset), axis=0)
        dset[old_len:] = data

        return len(dset)
    elif len(data):
        # If not, create it
        maxshape = (None, data.shape[1], data.shape[2])
        dset = f.create_dataset("clips", data=data, maxshape=maxshape, chunks=True)
        return len(dset)

    f.close()

#folder_path = '/media/seb/ee7d6f3e-3390-444a-b0b3-131b80f2a7f8/datasets/dataset_cmd/raw_footage/'
#files_arr = glob.glob(os.path.join(folder_path, '*'))

folder_path = '/media/seb/ee7d6f3e-3390-444a-b0b3-131b80f2a7f8/datasets/dataset_wildlife/raw_footage/'
#with open('files_list_update3.txt', 'r') as f:
#    files_arr = f.read().split('\n')
files_arr = np.load('all_files_keep_new_new.npy')

for file_idx, file_name in enumerate(files_arr):
    print('\nFile {}/{}, "{}"'.format(file_idx+1, len(files_arr), file_name))
    
    file_name = os.path.join(folder_path, file_name)

    try:
        data = get_array_from_video(file_name, n_frames=40)
        print('\t... Processed video into {} training examples {}'.format(data.shape[0], data.shape))
    except Exception as e:
        print(e)
        continue

    if not len(data):
        print('\t... No clips detected, skipping file')
        continue

    crop_size, x_crops, y_crops = (36, 36), 35, 21
    data = process_data(data, crop_size, x_crops, y_crops, padding=0.1)
    print('\t... Cropped video with {} crops per frame'.format(x_crops*y_crops))

    data = np.transpose(data, (0, 3, 1, 2)) 
    total_clips = data.shape[0]
    print('\t... Transposed data into {}'.format(data.shape))

    data = reshape_data(data)
    print('\t... Reshaped data into {}'.format(data.shape))

    data = reject_low_motion_clips(data)
    if len(data):
        print('\t... Removed low motion clips (kept {}/{})'.format(len(data), total_clips))
    else:
        print('\t... No high motion clips detected, skipping file')
        continue

    data = normalize_data(data)
    print('\t... Normalized data')

    data = data.astype(np.float16)

    np.random.shuffle(data)
    print('\t... Shuffled data')

    path_name =  f"/media/seb/Elements/datasets_wildlife_momotion/hierarchical_wildlife_36x36_1dg_big.hdf"
    dset_len = save_data_hdf5(data, path_name)
    print("\t... Saved data as '{}', {} clips total".format(path_name, dset_len))
