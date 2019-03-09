import torch
import torch.utils.data
import os
import librosa

DATASET_ROOT = '/home/cly/datacenter/songs'

class SongSegs(torch.utils.data.Dataset):

    def __init__(self, singer):
        self.singer_root = os.path.join(DATASET_ROOT, singer)
        self.song_segs = os.listdir(self.singer_root)
        self.cache = {}

    def __len__(self):
        # return 50000 #for debug
        return len(self.song_segs)

    def __getitem__(self, index):
        # index = index % 2 + 10000 #only 2 samples for debug...
        path = os.path.join(self.singer_root, self.song_segs[index])
        if self.cache.get(path) is None:
            self.cache[path], _ = librosa.load(path, 20000) #return sample && sr
        return self.cache[path]
