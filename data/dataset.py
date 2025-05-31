import os
import numpy as np
import torch
import torch.utils.data

class DatasetUnsupervised (torch.utils.data.Dataset):
    def __init__ (self, path, mmap=False):
        self.path = path

        if mmap:
            self.dataset = np.load(path, mmap_mode='r+')
        else:
            self.dataset = np.load(path)

    def __len__ (self):
        return len(self.dataset)

    def __getitem__ (self, i):
        clip = self.dataset[i]
        target = clip.copy()

        clip = torch.from_numpy(clip)
        clip = clip.type(torch.FloatTensor)

        target = torch.from_numpy(target)
        target = target.type(torch.FloatTensor)

        return clip, target

class DatasetUnsupervisedMulti (torch.utils.data.Dataset):
    def __init__ (self, paths, mmap=False):
        self.datasets = []

        for path_idx, path in enumerate(paths):
            print("Loading", path)

            if mmap:
                dataset = np.load(path, mmap_mode='r+')
            else:
                dataset = np.load(path)

            self.datasets.append(dataset)

    def __len__ (self):
        total_len = 0

        for dataset in self.datasets:
            total_len += len(dataset)

        return total_len

    def __getitem__ (self, i):
        dataset_len = len(self.datasets[0])
        dataset_i = i // dataset_len
        window_i = i - dataset_len*dataset_i

        window = self.datasets[dataset_i][window_i]
        window = torch.from_numpy(window)
        window = window.type(torch.FloatTensor)

        return window, window

    
def data_loader (dataset_path, split, batch_size=256, num_workers=1, pin_memory=True, shuffle=True, mmap=False):
    if type(dataset_path) == list:
        full_dataset = DatasetUnsupervisedMulti(dataset_path, mmap)
    else:
        full_dataset = DatasetUnsupervised(dataset_path, mmap)

    len_full_dataset = len(full_dataset) # Or desired total size to use

    train_size = int(40/44 * len_full_dataset)
    test_size = len_full_dataset - train_size

    all_idxs = range(0, len_full_dataset)
    test_idxs = np.linspace(0, len_full_dataset-1, test_size, dtype=int)
    train_idxs = list(set(all_idxs).difference(test_idxs))

    if split == 'train':
        dataset_split = torch.utils.data.Subset(full_dataset, train_idxs)
    elif split == 'validation':
        dataset_split = torch.utils.data.Subset(full_dataset, test_idxs)
    elif split == 'full':
        dataset_split = torch.utils.data.Subset(full_dataset, all_idxs)
    else:
        raise NotImplementedError

    data_loader = torch.utils.data.DataLoader(
        dataset_split, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    print("Dataset ({}) length: {}".format(split, len(dataset_split)))

    return data_loader
