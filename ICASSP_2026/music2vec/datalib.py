import os
import glob
import torch
import torchaudio
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler
from transformers import Wav2Vec2Processor
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import scipy.signal as signal
import random

class FakeMusicCapsDataset(Dataset):
    def __init__(self, file_paths, labels, sr=16000, target_duration=10.0, augment=True):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.target_samples = int(target_duration * sr) 
        self.augment = augment
    def __len__(self):
        return len(self.file_paths)

    def augment_audio(self, y, sr):
        if isinstance(y, torch.Tensor):
            y = y.numpy() 
        if random.random() < 0.5:
            rate = random.uniform(0.8, 1.2)
            y = librosa.effects.time_stretch(y=y, rate=rate)
        if random.random() < 0.5:
            n_steps = random.randint(-2, 2)
            y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps) 
        if random.random() < 0.5:
            noise_level = np.random.uniform(0.001, 0.005)
            y = y + np.random.normal(0, noise_level, y.shape)
        if random.random() < 0.5:
            gain = np.random.uniform(0.9, 1.1)
            y = y * gain
        return torch.tensor(y, dtype=torch.float32)  

    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(audio_path)
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)(waveform)
        waveform = waveform.mean(dim=0)  
        current_samples = waveform.shape[0]

        if label == 0:
            waveform = self.augment_audio(waveform, self.sr)
        if label == 1:
            waveform = self.highpass_filter(waveform, self.sr)
            waveform = self.augment_audio(waveform, self.sr)

        if current_samples > self.target_samples:
            waveform = waveform[:self.target_samples] 
        elif current_samples < self.target_samples:
            pad_length = self.target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))  

        # waveform = waveform.squeeze(0)
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform, dtype=torch.float32)

        return waveform.unsqueeze(0), torch.tensor(label, dtype=torch.long)  

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

    def preprocess_audio(audio_path, target_sr=16000, max_length=160000):
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

        waveform = waveform.mean(dim=0).unsqueeze(0)  

        current_samples = waveform.shape[1]
        if current_samples > max_length:
            start_idx = (current_samples - max_length) // 2
            waveform = waveform[:, start_idx:start_idx + max_length]
        elif current_samples < max_length:
            pad_length = max_length - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

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
