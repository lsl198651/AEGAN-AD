import os
import numpy as np
from torch.utils.data import Dataset
import torch
import utils
import torchaudio.compliance.kaldi as ta_kaldi


class train_dataset(Dataset):
    def __init__(self, features, targets):
        '''Dataset for training purpose. Output one segment at a time.

        Args:
            param (dict): hyper parameters stored in config.yaml
        '''
        self.data = torch.tensor(features)
        self.traget = torch.tensor(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):  # output one segment at a time
        dataitem = self.data[idx]
        targetitem = self.traget[idx]
        # logmel = ta_kaldi.fbank(dataitem, num_mel_bins=128, sample_frequency=16000,
        #                         frame_length=25, frame_shift=10)
        return dataitem, targetitem


class test_dataset(Dataset):
    def __init__(self, features, targets):
        '''Dataset for testing purpose. Output segments of a clip at a time.

        Args:
            param (dict): hyper parameters stored in config.yaml
            set_type (str): 'dev' or 'eval'. Two test sets are not mixed together.
            data_type (str, optional): use train data or test data for validation. Defaults to 'test'.
        '''
        self.data = torch.tensor(features)
        self.traget = torch.tensor(targets)

    def __len__(self):  # number of clips
        return len(self.data)

    def __getitem__(self, idx):  # output segments of a clip at a time
        dataitem = self.data[idx]
        targetitem = self.traget[idx]
        logmel = ta_kaldi.fbank(
            dataitem.long(), num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10, window_type='hamming')
        return logmel, targetitem

    # def get_sec(self):
    #     return np.unique(self.all_attri[:, 0]).tolist()

    # def get_clip_name(self):
    #     return list(map(lambda f: os.path.basename(f), self.set_clip_addr[self.set_type]))
