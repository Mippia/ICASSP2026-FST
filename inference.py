import os
from pathlib import Path
import json
import numpy as np
import torch
from typing import List, Tuple, Optional
import pytorch_lightning as pl
from model import MusicAudioClassifier
import argparse
import torch
import torchaudio
import scipy.signal as signal
from typing import Dict, List
from networks import MERT_AudioCNN
from preprocess import get_segments_from_wav, find_optimal_segment_length




def load_audio(audio_path: str, sr: int = 24000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    오디오 파일을 불러와 세그먼트로 분할합니다.
    고정된 길이의 세그먼트를 최대 48개 추출하고, 부족한 경우 패딩을 추가합니다.
    
    Args:
        audio_path: 오디오 파일 경로
        sr: 목표 샘플링 레이트 (기본값 24000)
    
    Returns:
        Tuple containing:
        - 오디오 파형이 담긴 텐서 (48, 1, 240000)
        - 패딩 마스크 텐서 (48), True = 패딩, False = 실제 오디오
    """
    
    beats, downbeats = get_segments_from_wav(audio_path)
    optimal_length, cleaned_downbeats = find_optimal_segment_length(downbeats)
    waveform, sample_rate = torchaudio.load(audio_path)
    # 데이터 타입을 float32로 변환
    waveform = waveform.to(torch.float32)

    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(sample_rate, sr)
        waveform = resampler(waveform)

    # 모노로 변환 (필요한 경우)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 120000 샘플 = 5초 @ 24kHz
    fixed_samples = 240000

    # 5초 길이의 무음(silence) 패딩 생성
    if waveform.shape[1]<= 240000:
        padding = torch.zeros(1, 120000, dtype=torch.float32)
        # 원본 오디오 뒤에 패딩 추가
        waveform = torch.cat([waveform, padding], dim=1)

# 각 downbeat에서 시작하는 segment 생성
    segments = []
    for i, start_time in enumerate(cleaned_downbeats):
        # 시작 샘플 인덱스 계산
        start_sample = int(start_time * sr)
        
        # 끝 샘플 인덱스 계산 (시작 지점 + 고정 길이)
        end_sample = start_sample + fixed_samples
        
        # 파일 끝을 넘어가는지 확인
        if end_sample > waveform.size(1):
            continue
        
        # 정확히 fixed_samples 길이의 세그먼트 추출
        segment = waveform[:, start_sample:end_sample]
        # 하이패스 필터 적용 - 채널 차원 유지
        #filtered = torch.tensor(highpass_filter(segment.squeeze().numpy(), sr)).unsqueeze(0) # 이거 모르겠다야..? 다양한 전처리 후 inference해보는거도 괜찮겠네
        filtered = torch.tensor(segment.squeeze().numpy(), dtype=torch.float32).unsqueeze(0) # processor 안쓰네?
        #여기에 모델별 preprocess가 원래는 들어가는게 맞음. 
        segments.append(filtered)
        
        # 최대 48개 세그먼트만 사용
        if len(segments) >= 48:
            break
    
    # 세그먼트가 없는 경우 처리
    if not segments:
        return torch.zeros((1, 1, fixed_samples), dtype=torch.float32), torch.ones(1, dtype=torch.bool)
    
    # 스택하여 텐서로 변환 - (n_segments, 1, time_samples) 형태 유지
    stacked_segments = torch.stack(segments)
    
    # 실제 세그먼트 수 (패딩 아님)
    num_segments = stacked_segments.shape[0]
    
    # 패딩 마스크 생성 (False = 실제 오디오, True = 패딩)
    padding_mask = torch.zeros(48, dtype=torch.bool)
    
    # 48개 미만인 경우 패딩 추가
    if num_segments < 48:
        # 빈 세그먼트로 패딩 (zeros)
        padding = torch.zeros((48 - num_segments, 1, fixed_samples), dtype=torch.float32)
        stacked_segments = torch.cat([stacked_segments, padding], dim=0)
        
        # 패딩 마스크 설정 (True = 패딩)
        padding_mask[num_segments:] = True
    
    return stacked_segments, padding_mask

def run_inference(model, audio_segments: torch.Tensor, padding_mask: torch.Tensor, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
    """
    Run inference on audio segments.
    
    Args:
        model: The loaded model
        audio_segments: Preprocessed audio segments tensor (48, 1, 240000)
        device: Device to run inference on
    
    Returns:
        Dictionary with prediction results
    """
    model.eval()
    model.to(device)
    model = model.half()
    

    with torch.no_grad():
        # 데이터 형태 확인 및 조정
        # wav_collate_with_mask 함수와 일치하도록 처리
        if audio_segments.shape[1] == 1:  # (48, 1, 240000) 형태
            # 채널 차원 제거하고 배치 차원 추가
            audio_segments = audio_segments[:, 0, :].unsqueeze(0)  # (1, 48, 240000)
        else:
            audio_segments = audio_segments.unsqueeze(0)  # (1, 48, 768) # 사실 audio가 아니라 embedding segments일수도
        # 데이터를 half 타입으로 변환
        if padding_mask.dim() == 1:
            padding_mask = padding_mask.unsqueeze(0)  # [48] -> [1, 48]
        audio_segments = audio_segments.to(device)
        
        mask = padding_mask.to(device)
        
        # 추론 실행 (마스크 포함)
        outputs = model(audio_segments, mask)
        
        # 모델 출력 구조에 따라 처리
        if isinstance(outputs, dict):
            result = outputs
        else:
            # 단일 텐서인 경우 (로짓)
            logits = outputs.squeeze()
            prob = scaled_sigmoid(logits, scale_factor=1.0, linear_property=0.0).item()
            
            result = {
                "prediction": "Fake" if prob > 0.5 else "Real",
                "confidence": f"{max(prob, 1-prob)*100:.2f}",
                "fake_probability": f"{prob:.4f}",
                "real_probability": f"{1-prob:.4f}",
                "raw_output": logits.cpu().numpy().tolist()
            }
    
    return result

# Custom scaling function to moderate extreme sigmoid values
def scaled_sigmoid(x, scale_factor=0.2, linear_property=0.3):
    # Apply scaling to make sigmoid less extreme
    scaled_x = x * scale_factor
    # Combine sigmoid with linear component
    raw_prob = torch.sigmoid(scaled_x) * (1-linear_property) + linear_property * ((x + 25) / 50)
    # Clip to ensure bounds
    return torch.clamp(raw_prob, min=0.011, max=0.989)

# Apply the scaled sigmoid


def get_model(model_type, device):
    """Load the specified model."""
    if model_type == "MERT":
        #from model import MusicAudioClassifier
        #model = MusicAudioClassifier(input_dim=768, is_emb=True, mode = 'both', share_parameter = False).to(device)
        ckpt_file = 'checkpoints/step=075000-val_loss=0.0273-val_acc=0.9952.ckpt'#'mert_finetune_10.pth'
        model = MERT_AudioCNN.load_from_checkpoint(ckpt_file).to(device)
        model.eval()
        # model.load_state_dict(torch.load(ckpt_file, map_location=device))
        embed_dim = 768

    else:
        raise ValueError(f"Unknown model type: {model_type}")
   
    
    model.eval()
    return model, embed_dim
    

def inference(audio_path):
    backbone_model, input_dim = get_model('MERT', 'cuda')
    segments, padding_mask = load_audio(audio_path, sr=24000)
    segments = segments.to('cuda').to(torch.float32)
    padding_mask = padding_mask.to('cuda').unsqueeze(0)    
    logits,embedding = backbone_model(segments.squeeze(1))

    # 모델 로드 부분 추가
    model = MusicAudioClassifier.load_from_checkpoint(
        checkpoint_path = 'checkpoints/EmbeddingModel_MERT_768_2class-epoch=0003-val_loss=0.0055-val_acc=0.9987-val_f1=0.9983-val_precision=0.9989-val_recall=0.9978.ckpt',
        input_dim=input_dim, 
        backbone = 'fusion_segment_transformer'
        #emb_model=backbone_model
        is_emb = True,
    )
    
    
    # Run inference
    print(f"Segments shape: {segments.shape}")
    print("Running inference...")
    results = run_inference(model, embedding, padding_mask, 'cuda')
    
    # 결과 출력
    print(f"Results: {results}")


    
    return results

if __name__ == "__main__":
    inference("picachu.mp3")

