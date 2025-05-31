import glob
import numpy as np
import scipy.stats as stats

# Paths to processed neural data
neural_paths = {
    'VISp' : glob.glob(''),
    'VISl' : glob.glob(''),
    'VISrl': glob.glob('')
}

model_paths = {
}

def get_chunked_neural_response_from_path (p, chunk_size=5):
    chunked_response_cells = []

    data = np.load(p, allow_pickle=True).item()

    for cell in data.values():
        chunked_response = []
        for d in cell['spikes_mean'].values():
            for i in range(0, d.shape[0], chunk_size):
                chunked_response.append(d[i:i+chunk_size].mean(axis=0))
        chunked_response_cells.append(chunked_response)
    return np.array(chunked_response_cells).T

def get_chunked_response_from_path (p, chunk_size=5):
    chunked_response = []

    data = np.load(p, allow_pickle=True).item()
    for d in data.values():
        for i in range(0, d.shape[0], chunk_size):
            chunked_response.append(
                np.nanmean(d[i:i+chunk_size], axis=0)
            )

    return np.array(chunked_response)

def get_RSM (responses, log=True):
    M_clips = responses.shape[0]
    RSM = np.zeros((M_clips, M_clips))

    for idx_a, clip_a in enumerate(responses):
        for idx_b, clip_b in enumerate(responses):
            if len(set(clip_a))==1 or len(set(clip_b))==1:
                RSM[idx_a, idx_b] = 0
            else:
                RSM[idx_a, idx_b]  = stats.pearsonr(clip_a, clip_b)[0]

    return RSM

def get_RSM_similarity (RSM_a, RSM_b):
    RSM_a_flat = RSM_a[np.triu_indices_from(RSM_a, k = 1)]
    RSM_b_flat = RSM_b[np.triu_indices_from(RSM_b, k = 1)]

    return stats.spearmanr(RSM_a_flat, RSM_b_flat)[0]

if True:
    save_model_data = {}

    for model_name, path in model_paths.items():
        print('Starting', model_name)

        data = get_chunked_response_from_path(path)
        print(data.shape)
        RSM  = get_RSM(data)
        print(RSM.shape)

        save_model_data[model_name] = RSM
        np.save('./RSA_data_model_RSM.npy', save_model_data)

if True:
    save_neural_data = {}
    for brain_area, paths in neural_paths.items():    
        save_neural_data[brain_area] = {}

        for path in paths:
            sess = int(path.split('/')[-1].split('_')[0])
            print('Starting', brain_area, sess)

            data = get_chunked_neural_response_from_path(path)
            print(data.shape)
            RSM  = get_RSM(data)
            print(RSM.shape)

            save_neural_data[brain_area][sess] = RSM
            np.save('./RSA_data_neural_RSM.npy', save_neural_data)
