import numpy as np
import pandas as pd
import librosa
import sklearn
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, left_path, right_path, label=None, mode='train'):
        self.left_path = left_path
        self.right_path = right_path
        self.max_len = 6000
        self.sr = 16000
        self.n_mfcc = 100
        self.n_fft = 400
        self.hop_length = 160
        self.mode = mode

        if self.mode == 'train':
            self.label = label

    def __len__(self):
        return len(self.left_path)
    
    def wav2image_tensor(self, path) :
        audio, sr = librosa.load(path, sr=self.sr)
        mfcc = librosa.feature.mfcc(audio, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
        pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))
        padded_mfcc = pad2d(mfcc, self.max_len).reshape(1, self.n_mfcc, self.max_len) #채널 추가
        padded_mfcc = torch.tensor(padded_mfcc, dtype=torch.float)

        return padded_mfcc

    def __getitem__(self, i):
        left_path = self.left_path[i]
        right_path = self.right_path[i]
        
        left_padded_mfcc = self.wav2image_tensor(left_path)
        right_padded_mfcc = self.wav2image_tensor(right_path)
        
        padded_mfcc = torch.cat([left_padded_mfcc, right_padded_mfcc], dim=0)

        if self.mode == 'train':
            label = self.label[i]
            return {
                'X': padded_mfcc,
                'Y': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'X': padded_mfcc
            }

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 100, 6000, 2)
        #    Conv     -> (?, 100, 6000, 32)
        #    Pool     -> (?, 50, 3000, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(2, 32, kernel_size=3, stride=1, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 50, 3000, 32)
        #    Conv      ->(?, 50, 3000, 64)
        #    Pool      ->(?, 25, 1500, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        # 전결합층 25x200x64 inputs -> 2 outputs
        self.fc = torch.nn.Linear(25*1500*64, 2, bias=True)

        torch.nn.init.xavier_uniform_(self.fc.weight) # fc 가중치 초기화

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # Flatten
        out = self.fc(out)
        return out