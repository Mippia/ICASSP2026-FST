import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import torch
import torch.nn as nn
import soundfile as sf

from networks import audiocnn, AudioCNNWithViTDecoder, AudioCNNWithViTDecoderAndCrossAttention


def highpass_filter(y, sr, cutoff=500, order=5):
    """High-pass filter to remove low frequencies below `cutoff` Hz."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y_filtered = signal.lfilter(b, a, y)
    return y_filtered

def plot_combined_visualization(y_original, y_filtered, sr, save_path="combined_visualization.png"):
    """Plot waveform comparison and spectrograms in a single figure."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 1️⃣ Waveform Comparison
    time = np.linspace(0, len(y_original) / sr, len(y_original))
    axes[0].plot(time, y_original, label='Original', alpha=0.7)
    axes[0].plot(time, y_filtered, label='High-pass Filtered', alpha=0.7, linestyle='dashed')
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Waveform Comparison (Original vs High-pass Filtered)")
    axes[0].legend()
    
    # 2️⃣ Spectrogram - Original
    S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_original)), ref=np.max)
    img = librosa.display.specshow(S_orig, sr=sr, x_axis='time', y_axis='log', ax=axes[1])
    axes[1].set_title("Original Spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    
    # 3️⃣ Spectrogram - High-pass Filtered
    S_filt = librosa.amplitude_to_db(np.abs(librosa.stft(y_filtered)), ref=np.max)
    img = librosa.display.specshow(S_filt, sr=sr, x_axis='time', y_axis='log', ax=axes[2])
    axes[2].set_title("High-pass Filtered Spectrogram")
    fig.colorbar(img, ax=axes[2], format="%+2.0f dB")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def load_model(checkpoint_path, model_class, device):
    """Load a trained model from checkpoint."""
    model = model_class()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_audio(model, audio_tensor, device):
    """Make predictions using a trained model."""
    with torch.no_grad():
        audio_tensor = audio_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(audio_tensor)
        prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
    return prediction

# Load audio
audio_path = "/data/kym/AI Music Detection/audio/FakeMusicCaps/real/musiccaps/_RrA-0lfIiU.wav"  # Replace with actual file path
y, sr = librosa.load(audio_path, sr=None)
y_filtered = highpass_filter(y, sr, cutoff=500)

# Convert audio to tensor
audio_tensor = torch.tensor(librosa.feature.melspectrogram(y=y, sr=sr), dtype=torch.float).unsqueeze(0)
audio_tensor_filtered = torch.tensor(librosa.feature.melspectrogram(y=y_filtered, sr=sr), dtype=torch.float).unsqueeze(0)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_model = load_model("/data/kym/AI Music Detection/AudioCNN/ckpt/FakeMusicCaps/pretraining/best_model_audiocnn.pth", audiocnn, device)
highpass_model = load_model("/data/kym/AI Music Detection/AudioCNN/ckpt/FakeMusicCaps/500hz_Add_crossattn_decoder/best_model_AudioCNNWithViTDecoderAndCrossAttention.pth", AudioCNNWithViTDecoderAndCrossAttention, device)

# Predict
original_pred = predict_audio(original_model, audio_tensor, device)
highpass_pred = predict_audio(highpass_model, audio_tensor_filtered, device)

print(f"Original Model Prediction: {original_pred}")
print(f"High-pass Filter Model Prediction: {highpass_pred}")

# Generate combined visualization (all plots in one image)
plot_combined_visualization(y, y_filtered, sr, save_path="/data/kym/AI Music Detection/AudioCNN/hf_vis/rawvs500.png")
