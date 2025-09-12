import os
from pathlib import Path
import json
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
import pytorch_lightning as pl
from model import MusicAudioClassifier, MERT_AudioCAT
import argparse
import torchaudio
import scipy.signal as signal
from preprocess import get_segments_from_wav, find_optimal_segment_length

from rich.console import Console
from rich.table import Table

console = Console()

def print_results(results: dict):
    table = Table(title="🎵 AI-Generated Music Detection Results 🎵")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Prediction", results['prediction'])
    table.add_row("Confidence", f"{results['confidence']}%")
    table.add_row("Fake Probability", results['fake_probability'])
    table.add_row("Real Probability", results['real_probability'])

    console.print(table)

def load_audio(audio_path: str, sr: int = 24000) -> Tuple[torch.Tensor, torch.Tensor]:
    beats, downbeats = get_segments_from_wav(audio_path)
    optimal_length, cleaned_downbeats = find_optimal_segment_length(downbeats)
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(torch.float32)

    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(sample_rate, sr)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    fixed_samples = 240000  

    if waveform.shape[1] <= fixed_samples:
        padding = torch.zeros(1, fixed_samples, dtype=torch.float32)
        waveform = torch.cat([waveform, padding], dim=1)

    segments = []
    for start_time in cleaned_downbeats:
        start_sample = int(start_time * sr)
        end_sample = start_sample + fixed_samples
        if end_sample > waveform.size(1):
            continue
        segment = waveform[:, start_sample:end_sample]
        filtered = torch.tensor(segment.squeeze().numpy(), dtype=torch.float32).unsqueeze(0)
        segments.append(filtered)
        if len(segments) >= 48:
            break

    if not segments:
        return torch.zeros((1, 1, fixed_samples), dtype=torch.float32), torch.ones(1, dtype=torch.bool)

    stacked_segments = torch.stack(segments)
    num_segments = stacked_segments.shape[0]
    padding_mask = torch.zeros(48, dtype=torch.bool)

    if num_segments < 48:
        padding = torch.zeros((48 - num_segments, 1, fixed_samples), dtype=torch.float32)
        stacked_segments = torch.cat([stacked_segments, padding], dim=0)
        padding_mask[num_segments:] = True

    return stacked_segments, padding_mask


def run_inference(model, audio_segments: torch.Tensor, padding_mask: torch.Tensor,
                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
    model.eval()
    model.to(device)
    model = model.half()

    with torch.no_grad():
        if audio_segments.shape[1] == 1:
            audio_segments = audio_segments[:, 0, :].unsqueeze(0)  # (1, 48, 240000)
        else:
            audio_segments = audio_segments.unsqueeze(0)

        if padding_mask.dim() == 1:
            padding_mask = padding_mask.unsqueeze(0)

        audio_segments = audio_segments.to(device)
        mask = padding_mask.to(device)

        outputs = model(audio_segments, mask)

        if isinstance(outputs, dict):
            result = outputs
        else:
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


def scaled_sigmoid(x, scale_factor=0.2, linear_property=0.3):
    scaled_x = x * scale_factor
    raw_prob = torch.sigmoid(scaled_x) * (1-linear_property) + linear_property * ((x + 25) / 50)
    return torch.clamp(raw_prob, min=0.011, max=0.989)


def get_model(model_type, device):
    if model_type == "MERT":
        ckpt_file = ""  # TODO: Download Stage-1 checkpoint and set path here
        model = MERT_AudioCAT.load_from_checkpoint(ckpt_file).to(device)
        model.eval()
        embed_dim = 768
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model, embed_dim

def inference(audio_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone_model, input_dim = get_model('MERT', device)
    segments, padding_mask = load_audio(audio_path, sr=24000)
    segments = segments.to(device).to(torch.float32)
    padding_mask = padding_mask.to(device).unsqueeze(0)

    logits, embedding = backbone_model(segments.squeeze(1))

    model = MusicAudioClassifier.load_from_checkpoint(
        checkpoint_path='', # TODO: Download Stage-2 checkpoint and set path here
        input_dim=input_dim,
        backbone='fusion_segment_transformer',
        is_emb=True,
    )

    print("Running inference...")
    results = run_inference(model, embedding, padding_mask, device)
    print_results(results)
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-generated music detection")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file to analyze")
    args = parser.parse_args()

    inference(args.audio)
