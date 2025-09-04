import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# 🔹 오디오 파일 로드
file_real = "/path/to/real_audio.wav"  # Real 오디오 경로
file_fake = "/path/to/generative_audio.wav"  # AI 생성 오디오 경로

def plot_spectrogram(audio_file, title):
    y, sr = librosa.load(audio_file, sr=16000)  # 샘플링 레이트 16kHz
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)  # STFT 변환

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.ylim(4000, 16000)  # 4kHz 이상 고주파 영역만 표시
    plt.show()

# 🔹 Real vs Generative Spectrogram 비교
plot_spectrogram(file_real, "Real Audio Spectrogram (4kHz+)")
plot_spectrogram(file_fake, "Generative Audio Spectrogram (4kHz+)")

