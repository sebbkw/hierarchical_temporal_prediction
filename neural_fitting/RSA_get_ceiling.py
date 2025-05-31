import glob
import numpy as np
import scipy.stats as stats

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

def get_RSM (responses, log=True):
    M_clips = responses.shape[0]
    RSM = np.zeros((M_clips, M_clips))
    
    for idx_a, clip_a in enumerate(responses):
        for idx_b, clip_b in enumerate(responses):                                    
            RSM[idx_a, idx_b]  = stats.pearsonr(clip_a, clip_b)[0]
                        
    return RSM

def get_RSM_similarity (RSM_a, RSM_b):        
    RSM_a_flat = RSM_a[np.triu_indices_from(RSM_a, k = 1)]
    RSM_b_flat = RSM_b[np.triu_indices_from(RSM_b, k = 1)]

    return stats.spearmanr(RSM_a_flat, RSM_b_flat)[0]

# Paths to processed neural data
neural_paths = {
    'VISl' : glob.glob(''),
    'VISrl': glob.glob(''),
    'VISp' : glob.glob('')
}

repeats = 100

save_ceiling_data = {}
for brain_area, paths in neural_paths.items():    
    save_ceiling_data[brain_area] = {}
    
    for path in paths:
        sess = int(path.split('/')[-1].split('_')[0])
        print('Starting', brain_area, sess)

        data = get_chunked_neural_response_from_path(path)
        sim_arr = []

        for i in range(repeats):
            print('\tStarting repeat', i, end='')

            shuffled_idxs = np.random.choice(data.shape[1], data.shape[1], replace=False)
            half_len      = len(shuffled_idxs)//2

            data_shuffled = data[:, shuffled_idxs]
            data_a = data_shuffled[:, :half_len]
            data_b = data_shuffled[:, half_len:]

            RSM_a = get_RSM(data_a)
            RSM_b = get_RSM(data_b)
            sim   = get_RSM_similarity(RSM_a, RSM_b)

            sim_arr.append(sim)
            print(f' (CC={sim:.3g})')


        save_ceiling_data[brain_area][sess] = sim_arr
        np.save('./model_data/RSA_data_ceiling.npy', save_ceiling_data)
