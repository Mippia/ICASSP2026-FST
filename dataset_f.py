import os
import glob
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
import scipy.signal as signal
import scipy.signal
import random
    def __init__(self, file_paths, labels, sr=16000, target_duration=10.0):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.target_samples = int(target_duration * sr)  # Fixed length: 10 seconds
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

    def __len__(self):
        return len(self.file_paths)
    
    def pre_emphasis(self, x, alpha=0.97):
        return np.append(x[0], x[1:] - alpha * x[:-1])
    
        
    # 시간 조절(Time Stretch), 이퀄라이저 조정(EQ), 리버브 추가
    def augment_audio(self, y, sr):
        if isinstance(y, torch.Tensor):
            y = y.numpy()  # Tensor → Numpy 변환

        if random.random() < 0.5:  # 시간 조절 (Time Stretch)
            rate = random.uniform(0.8, 1.2)
            y = librosa.effects.time_stretch(y=y, rate=rate)

        if random.random() < 0.5:  # 피치 시프트 (Pitch Shift)
            n_steps = random.randint(-2, 2)
            y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps) 

        if random.random() < 0.5:  # 화이트 노이즈 추가 (White Noise Addition)
            noise_level = np.random.uniform(0.001, 0.005)
            y = y + np.random.normal(0, noise_level, y.shape)


        return torch.tensor(y, dtype=torch.float32)  # 다시 Tensor로 변환
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(audio_path)

        target_sr = self.processor.sampling_rate  

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        waveform = waveform.mean(dim=0).squeeze(0) 
        if label == 0:
            waveform = self.augment_audio(waveform, self.sr)
        if label == 1:
            waveform = self.highpass_filter(waveform, self.sr)
            # waveform = self.pre_emphasis(waveform)
            waveform = self.augment_audio(waveform, self.sr)
        # if label == 1:
        #     waveform = self.pre_emphasis(waveform)
        #     waveform = torch.tensor(waveform, dtype=torch.float32)


        current_samples = waveform.shape[0]
        if current_samples > self.target_samples:
            waveform = waveform[:self.target_samples]  # Truncate
        elif current_samples < self.target_samples:
            pad_length = self.target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))  # Pad

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()  # Tensor일 경우에만 변환
        print(waveform.shape)
        inputs = self.processor(waveform, sampling_rate=target_sr, return_tensors="pt", padding=True)
        print(inputs["input_values"].shape)

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


def collect_files(base_path):
    real_files = glob.glob(os.path.join(base_path, "real", "**", "*.wav"), recursive=True)
    gen_files = glob.glob(os.path.join(base_path, "generative", "**", "*.wav"), recursive=True)
    real_labels = [0] * len(real_files)
    gen_labels = [1] * len(gen_files)
    return real_files + gen_files, real_labels + gen_labels
