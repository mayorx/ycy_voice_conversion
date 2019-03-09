import torch
import torch.utils.data
import os
import librosa
import numpy as np

DATASET_ROOT = '/home/cly/datacenter/songs'
LENGTH = 1000

class SongSegs(torch.utils.data.Dataset):

    def __init__(self, singer, len = 100000):
        # self.singer_root = os.path.join(DATASET_ROOT, singer)
        # self.song_segs = os.listdir(self.singer_root)
        self.raw_data = np.fromfile(os.path.join(DATASET_ROOT, '{}.dat').format(singer), np.float32)
        self.len = len
        # self.cache = {}

    def __len__(self):
        return self.len
        # return len(self.song_segs)

    def __getitem__(self, index):
        st = np.random.randint(0, len(self.raw_data) - LENGTH)
        return self.raw_data[st:st + LENGTH]
        # index = index % 2 + 10000 #only 2 samples for debug...
        # path = os.path.join(self.singer_root, self.song_segs[index])
        # if self.cache.get(path) is None:
        #     self.cache[path], _ = librosa.load(path, 20000) #return sample && sr
        # return self.cache[path]
