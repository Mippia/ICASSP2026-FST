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
from transformers import Wav2Vec2FeatureExtractor
import scipy.signal as signal
import scipy.signal
# class FakeMusicCapsDataset(Dataset):
#     def __init__(self, file_paths, labels, sr=16000, target_duration=10.0):
#         self.file_paths = file_paths
#         self.labels = labels
#         self.sr = sr
#         self.target_samples = int(target_duration * sr)  # Fixed length: 5 seconds

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         audio_path = self.file_paths[idx]
#         label = self.labels[idx]

#         waveform, sr = torchaudio.load(audio_path)
#         waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)(waveform)
#         waveform = waveform.mean(dim=0)  # Convert to mono
#         waveform = waveform.squeeze(0)


#         current_samples = waveform.shape[0]

#         # **Ensure waveform is exactly `target_samples` long**
#         if current_samples > self.target_samples:
#             waveform = waveform[:self.target_samples]  # Truncate if too long
#         elif current_samples < self.target_samples:
#             pad_length = self.target_samples - current_samples
#             waveform = torch.nn.functional.pad(waveform, (0, pad_length))  # Pad if too short

#         return waveform.unsqueeze(0), torch.tensor(label, dtype=torch.long)  # Ensure 2D shape (1, target_samples)

class FakeMusicCapsDataset(Dataset):
    def __init__(self, file_paths, labels, sr=16000, target_duration=10.0):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.target_samples = int(target_duration * sr)  # Fixed length: 10 seconds
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

    def __len__(self):
        return len(self.file_paths)

    def highpass_filter(self, y, sr, cutoff=500, order=5): 
            if isinstance(sr, np.ndarray):
                # print(f"[ERROR] sr is an array, taking mean value. Original sr: {sr}")
                sr = np.mean(sr)  
            if not isinstance(sr, (int, float)):
                raise ValueError(f"[ERROR] sr must be a number, but got {type(sr)}: {sr}")
            # print(f"[DEBUG] Highpass filter using sr={sr}, cutoff={cutoff}")  
            if sr <= 0:
                raise ValueError(f"Invalid sample rate: {sr}. It must be greater than 0.")
            nyquist = 0.5 * sr
            # print(f"[DEBUG] Nyquist frequency={nyquist}")  
            if cutoff <= 0 or cutoff >= nyquist:
                print(f"[WARNING] Invalid cutoff frequency {cutoff}, adjusting...")
                cutoff = max(10, min(cutoff, nyquist - 1))  
            normal_cutoff = cutoff / nyquist
            # print(f"[DEBUG] Adjusted cutoff={cutoff}, normal_cutoff={normal_cutoff}")  
            b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
            y_filtered = signal.lfilter(b, a, y)
            return y_filtered
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(audio_path)

        target_sr = self.processor.sampling_rate  

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        waveform = waveform.mean(dim=0).squeeze(0)  # [Time]

        if label == 1:
            waveform = self.highpass_filter(waveform, self.sr)

        current_samples = waveform.shape[0]
        if current_samples > self.target_samples:
            waveform = waveform[:self.target_samples]  # Truncate
        elif current_samples < self.target_samples:
            pad_length = self.target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))  # Pad

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()  # Tensor일 경우에만 변환

        inputs = self.processor(waveform, sampling_rate=target_sr, return_tensors="pt", padding=True)

        return inputs["input_values"].squeeze(0), torch.tensor(label, dtype=torch.long)  # [1, time] → [time]

    @staticmethod
    def collate_fn(batch, target_samples=16000 * 10):

        inputs, labels = zip(*batch)  # Unzip batch

        processed_inputs = []
        for waveform in inputs:
            current_samples = waveform.shape[0]

            if current_samples > target_samples:
                start_idx = (current_samples - target_samples) // 2
                cropped_waveform = waveform[start_idx:start_idx + target_samples]
            else:
                pad_length = target_samples - current_samples
                cropped_waveform = torch.nn.functional.pad(waveform, (0, pad_length))

            processed_inputs.append(cropped_waveform)

        processed_inputs = torch.stack(processed_inputs)  # [batch, target_samples]
        labels = torch.tensor(labels, dtype=torch.long)  # [batch]

        return processed_inputs, labels
    
    def preprocess_audio(audio_path, target_sr=16000, max_length=160000):
        """
        오디오를 모델 입력에 맞게 변환
        - target_sr: 16kHz로 변환
        - max_length: 최대 길이 160000 (10초)
        """
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

        # Convert to mono
        waveform = waveform.mean(dim=0).unsqueeze(0)  # (1, sequence_length)

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

# Closed Test: FakeMusicCaps 데이터셋 사용
real_files = glob.glob(os.path.join(DATASET_PATH, "real", "**", "*.wav"), recursive=True)
gen_files = glob.glob(os.path.join(DATASET_PATH, "generative", "**", "*.wav"), recursive=True)

# Open Set Test: SUNOCAPS_PATH 데이터 포함
open_real_files = real_files + glob.glob(os.path.join(SUNOCAPS_PATH, "real", "**", "*.wav"), recursive=True)
open_gen_files = gen_files + glob.glob(os.path.join(SUNOCAPS_PATH, "generative", "**", "*.wav"), recursive=True)

real_labels = [0] * len(real_files)
gen_labels = [1] * len(gen_files)

open_real_labels = [0] * len(open_real_files)
open_gen_labels = [1] * len(open_gen_files)

# Closed Train, Val
real_train, real_val, real_train_labels, real_val_labels = train_test_split(real_files, real_labels, test_size=0.2, random_state=42)
gen_train, gen_val, gen_train_labels, gen_val_labels = train_test_split(gen_files, gen_labels, test_size=0.2, random_state=42)

train_files = real_train + gen_train
train_labels = real_train_labels + gen_train_labels
val_files = real_val + gen_val
val_labels = real_val_labels + gen_val_labels

# Closed Set Test용 데이터셋
closed_test_files = real_files + gen_files
closed_test_labels = real_labels + gen_labels

# Open Set Test용 데이터셋 
open_test_files = open_real_files + open_gen_files
open_test_labels = open_real_labels + open_gen_labels

# Oversampling 적용
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
train_files_resampled, train_labels_resampled = ros.fit_resample(np.array(train_files).reshape(-1, 1), train_labels)

train_files = train_files_resampled.reshape(-1).tolist()
train_labels = train_labels_resampled

print(f"📌 Train Original FAKE: {len(gen_train)}")
print(f"📌 Train set (Oversampled) - REAL: {sum(1 for label in train_labels if label == 0)}, "
      f"FAKE: {sum(1 for label in train_labels if label == 1)}, Total: {len(train_files)}")
print(f"📌 Validation set - REAL: {len(real_val)}, FAKE: {len(gen_val)}, Total: {len(val_files)}")
