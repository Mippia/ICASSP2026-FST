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
# Oversampling Lib
from imblearn.over_sampling import RandomOverSampler

class FakeMusicCapsDataset(Dataset):
    def __init__(self, file_paths, labels, feat_type=['mel'], sr=16000, n_mels=64, target_duration=10.0, augment=True, augment_real=True):
        self.file_paths = file_paths
        self.labels = labels
        self.feat_type = feat_type
        self.sr = sr
        self.n_mels = n_mels
        self.target_duration = target_duration
        self.target_samples = int(target_duration * sr)
        self.augment = augment
        self.augment_real = augment_real  


    def pre_emphasis(self, x, alpha=0.97):
        return np.append(x[0], x[1:] - alpha * x[:-1])

    def highpass_filter(self, y, sr, cutoff=1000, order=5):
        nyquist = 0.5 * sr
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return signal.lfilter(b, a, y)

    def augment_audio(self, y, sr):
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

        return y

    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load and preprocess audio file.
        """
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, sr = librosa.load(audio_path, sr=self.sr, mono=True)  
        if label == 0:
            if self.augment_real:
                waveform = self.augment_audio(waveform, self.sr)
        if label == 1:
            waveform = self.highpass_filter(waveform, self.sr)
            waveform = self.augment_audio(waveform, self.sr)
        
        current_samples = waveform.shape[0]
        if current_samples > self.target_samples:
            start_idx = (current_samples - self.target_samples) // 2
            waveform = waveform[start_idx:start_idx + self.target_samples]
        elif current_samples < self.target_samples:
            waveform = np.pad(waveform, (0, self.target_samples - current_samples), mode='constant')


        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=self.sr, n_mels=self.n_mels, n_fft=1024, hop_length=256
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        log_mel_spec = np.expand_dims(log_mel_spec, axis=0)  
        mel_tensor = torch.tensor(log_mel_spec, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return mel_tensor, label_tensor

    def extract_feature(self, waveform, feat):
        """Extracts specified feature (mel, stft, cqt) from waveform."""
        try:
            if feat == 'mel':  
                mel_spec = librosa.feature.melspectrogram(y=waveform, sr=self.sr, n_mels=self.n_mels, n_fft=1024, hop_length=256)
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                return torch.tensor(log_mel_spec, dtype=torch.float).unsqueeze(0)
            elif feat == 'stft':  
                stft = librosa.stft(waveform, n_fft=512, hop_length=128, window="hann")
                logSTFT = np.log(np.abs(stft) + 1e-3)
                return torch.tensor(logSTFT, dtype=torch.float).unsqueeze(0)
            elif feat == 'cqt':  
                cqt = librosa.cqt(waveform, sr=self.sr, hop_length=128, bins_per_octave=24) 
                logCQT = np.log(np.abs(cqt) + 1e-3)
                return torch.tensor(logCQT, dtype=torch.float).unsqueeze(0)
            else:
                raise ValueError(f"[ERROR] Unsupported feature type: {feat}")
        except Exception as e:
            print(f"[ERROR] Feature extraction failed for {feat}: {e}")
            return None
        
    def highpass_filter(self, y, sr, cutoff=1000, order=5): 
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
    
def preprocess_audio(audio_path, sr=16000, n_mels=64, target_duration=10.0):
    try:
        waveform, _ = librosa.load(audio_path, sr=sr, mono=True)

        target_samples = int(target_duration * sr)
        if len(waveform) > target_samples:
            start_idx = (len(waveform) - target_samples) // 2
            waveform = waveform[start_idx:start_idx + target_samples]
        elif len(waveform) < target_samples:
            waveform = np.pad(waveform, (0, target_samples - len(waveform)), mode='constant')
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=256)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    except Exception as e:
        print(f"[ERROR] 전처리 실패: {audio_path} | 오류: {e}")
        return None


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
print(f"type(train_labels_resampled): {type(train_labels_resampled)}")

print(f"Train Org Fake: {len(gen_val)}")
print(f"Train set (Oversampled) - Real: {sum(1 for label in train_labels if label == 0)}, "
      f"Fake: {sum(1 for label in train_labels if label == 1)}, Total: {len(train_files)}")
print(f"Validation set - Real: {len(real_val)}, Fake: {len(gen_val)}, Total: {len(val_files)}")
print(f"Closed Test set - Real: {len(real_files)}, Fake: {len(gen_files)}, Total: {len(closed_test_files)}")
print(f"Open Test set - Real: {len(open_real_files)}, Fake: {len(open_gen_files)}, Total: {len(open_test_files)}")