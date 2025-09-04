import os
import glob
import random
import torch
import librosa
import numpy as np
import utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import scipy.signal as signal
import scipy.signal
from scipy.signal import butter, lfilter
import numpy as np
import scipy.signal as signal
import librosa
import torch
import random
from torch.utils.data import Dataset
import logging
import csv
import logging
import time
import numpy as np
import h5py
import torch
import torchaudio
from imblearn.over_sampling import RandomOverSampler
from networks import Wav2Vec2ForFakeMusic
from transformers import Wav2Vec2Processor
import torchaudio.transforms as T

class FakeMusicCapsDataset(Dataset):
    def __init__(self, file_paths, labels, sr=16000, target_duration=10.0):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.target_duration = target_duration
        self.target_samples = int(target_duration * sr)

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    def highpass_filter(self, y, sr, cutoff=500, order=5): 
            if isinstance(sr, np.ndarray):
                sr = np.mean(sr)  
            if not isinstance(sr, (int, float)):
                raise ValueError(f"[ERROR] sr must be a number, but got {type(sr)}: {sr}")
            if sr <= 0:
                raise ValueError(f"Invalid sample rate: {sr}. It must be greater than 0.")
            nyquist = 0.5 * sr
            if cutoff <= 0 or cutoff >= nyquist:
                print(f"[WARNING] Invalid cutoff frequency {cutoff}, adjusting...")
                cutoff = max(10, min(cutoff, nyquist - 1))  
            normal_cutoff = cutoff / nyquist
            b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
            y_filtered = signal.lfilter(b, a, y)
            return y_filtered
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(audio_path)
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)(waveform)

        waveform = waveform.squeeze(0)
        if label == 0:
            waveform = self.augment_audio(waveform, self.sr)
        if label == 1:
            waveform = self.highpass_filter(waveform, self.sr)

        current_samples = waveform.shape[0]
        if current_samples > self.target_samples:
            start_idx = (current_samples - self.target_samples) // 2
            waveform = waveform[start_idx:start_idx + self.target_samples]
        elif current_samples < self.target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_samples - current_samples))

        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  
        label = torch.tensor(label, dtype=torch.long)  

        return waveform, label

def preprocess_audio(audio_path, target_sr=16000, target_duration=10.0):
    waveform, sr = librosa.load(audio_path, sr=target_sr)  

    target_samples = int(target_duration * target_sr)
    current_samples = len(waveform)

    if current_samples > target_samples:
        waveform = waveform[:target_samples]  
    elif current_samples < target_samples:
        waveform = np.pad(waveform, (0, target_samples - current_samples))  

    waveform = torch.tensor(waveform).unsqueeze(0)
    return waveform

    
DATASET_PATH = "/data/kym/AI_Music_Detection/audio/FakeMusicCaps"
SUNOCAPS_PATH = "/data/kym/Audio/SunoCaps"  # Open Set 포함 데이터

real_files = glob.glob(os.path.join(DATASET_PATH, "real", "**", "*.wav"), recursive=True)
gen_files = glob.glob(os.path.join(DATASET_PATH, "generative", "**", "*.wav"), recursive=True)

open_real_files = real_files + glob.glob(os.path.join(SUNOCAPS_PATH, "real", "**", "*.wav"), recursive=True)
open_gen_files = gen_files + glob.glob(os.path.join(SUNOCAPS_PATH, "generative", "**", "*.wav"), recursive=True)

real_labels = [0] * len(real_files)
gen_labels = [1] * len(gen_files)

open_real_labels = [0] * len(open_real_files)
open_gen_labels = [1] * len(open_gen_files)

real_train, real_val, real_train_labels, real_val_labels = train_test_split(real_files, real_labels, test_size=0.2, random_state=42)
gen_train, gen_val, gen_train_labels, gen_val_labels = train_test_split(gen_files, gen_labels, test_size=0.2, random_state=42)

train_files = real_train + gen_train
train_labels = real_train_labels + gen_train_labels
val_files = real_val + gen_val
val_labels = real_val_labels + gen_val_labels

closed_test_files = real_files + gen_files
closed_test_labels = real_labels + gen_labels

open_test_files = open_real_files + open_gen_files
open_test_labels = open_real_labels + open_gen_labels

ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
train_files_resampled, train_labels_resampled = ros.fit_resample(np.array(train_files).reshape(-1, 1), train_labels)

train_files = train_files_resampled.reshape(-1).tolist()
train_labels = train_labels_resampled

print(f"Train Original FAKE: {len(gen_train)}")
print(f"Train set (Oversampled) - REAL: {sum(1 for label in train_labels if label == 0)}, "
      f"FAKE: {sum(1 for label in train_labels if label == 1)}, Total: {len(train_files)}")
print(f"Validation set - REAL: {len(real_val)}, FAKE: {len(gen_val)}, Total: {len(val_files)}")
