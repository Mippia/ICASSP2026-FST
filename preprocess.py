from beat_this.inference import File2Beats
import torchaudio
import torch
from pathlib import Path
import numpy as np
from collections import Counter
import os
import argparse
from tqdm import tqdm
import multiprocessing
import librosa
import gc

def get_segments_from_wav(wav_path, device="cuda", max_duration=300):
    """오디오 파일에서 비트와 다운비트를 추출합니다."""
    file2beats = File2Beats(checkpoint_path="final0", device=device, dbn=False)
    beats, downbeats = file2beats(wav_path)
    
    del file2beats
    torch.cuda.empty_cache() if device == "cuda" else None
    gc.collect()
    
    return beats, downbeats

def find_optimal_segment_length(downbeats, round_decimal=1, bar_length=4):
    """다운비트 간격들의 분포를 분석하여 최적의 4마디 길이와 정제된 다운비트 위치들을 반환합니다."""
    if len(downbeats) < 2:
        return 10.0, downbeats
    
    intervals = np.diff(downbeats)
    rounded_intervals = np.round(intervals, round_decimal)
    
    interval_counter = Counter(rounded_intervals)
    most_common_interval = interval_counter.most_common(1)[0][0]
    
    cleaned_downbeats = [downbeats[0]]
    
    for i in range(1, len(downbeats)):
        interval = rounded_intervals[i-1]
        if abs(interval - most_common_interval) <= most_common_interval * 0.1:
            cleaned_downbeats.append(downbeats[i])
    
    return float(most_common_interval * bar_length), np.array(cleaned_downbeats)

def process_audio_file(file_info, output_base_dir, device="cuda", max_duration=300, min_duration=30):
    """단일 오디오 파일을 처리하고 세그먼트를 추출합니다."""
    audio_file, relative_path, output_subdir = file_info
    
    # 출력 디렉토리 설정
    output_dir = Path(output_base_dir) / output_subdir
    file_seg_dir = output_dir / audio_file.stem
    
    # 이미 처리된 파일인지 체크
    if file_seg_dir.exists() and list(file_seg_dir.glob("segment_*.mp3")):
        return -1
    
    # 파일 크기 체크
    file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
    if file_size_mb > 100:
        return 0
    
    # 오디오 길이 체크
    info = torchaudio.info(str(audio_file))
    total_duration = info.num_frames / info.sample_rate
    
    if total_duration < min_duration:
        return 0
    
    beats, downbeats = get_segments_from_wav(str(audio_file), device=device, max_duration=max_duration)
    
    if beats is None or downbeats is None or len(downbeats) == 0:
        return 0
    
    optimal_length, cleaned_downbeats = find_optimal_segment_length(downbeats)
    
    file_seg_dir.mkdir(exist_ok=True, parents=True)
    
    sample_rate = info.sample_rate
    
    # 최대 길이 제한
    if total_duration > max_duration:
        total_duration = max_duration
    
    segments_count = 0
    
    # 각 다운비트에서 시작하는 세그먼트 생성
    for i, start_time in enumerate(cleaned_downbeats):
        end_time = start_time + optimal_length
        
        if end_time > total_duration:
            continue
        
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        
        segment, sr = torchaudio.load(
            str(audio_file), 
            frame_offset=start_frame,
            num_frames=end_frame - start_frame
        )
        
        # MP3로 세그먼트 저장 (320kbps)
        save_path = file_seg_dir / f"segment_{i}.mp3"
        torchaudio.save(
            str(save_path), 
            segment, 
            sr,
            backend = "sox",
            format="mp3",
            # # encoding="MP3",
            compression=320
        )
        segments_count += 1
        
        del segment
    
    torch.cuda.empty_cache() if device == "cuda" else None
    gc.collect()

    return segments_count

def process_file_wrapper(args):
    """multiprocessing용 래퍼 함수"""
    return process_audio_file(*args)

def segment_dataset(base_dir, output_base_dir, num_workers=4, device="cuda", max_duration=300, min_duration=30, labels=None):
    """멀티프로세싱을 사용한 세그먼트 추출"""
    base_path = Path(base_dir)
    
    stats = {
        "processed_files": 0,
        "extracted_segments": 0,
        "failed_files": 0,
        "skipped_files": 0,
    }
    
    # 레이블 설정
    if labels is None:
        labels = ["ai_cover", "real", "fake"]
    
    for label in labels:
        input_dir = base_path / label 
        
        if not input_dir.exists():
            continue
        
        # 해당 레이블 폴더에서 재귀적으로 오디오 파일 찾기
        audio_files = []
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
        
        for file_path in input_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                relative_path = file_path.relative_to(base_path)
                output_subdir = relative_path.parent
                audio_files.append((file_path, relative_path, output_subdir))
        
        if not audio_files:
            continue
        
        # 파일 크기별로 정렬
        audio_files.sort(key=lambda x: os.path.getsize(x[0]))
        
        # 멀티프로세싱으로 처리
        args_list = [(file_info, output_base_dir, device, max_duration, min_duration) for file_info in audio_files]
        
        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_file_wrapper, args_list), 
                              total=len(args_list), desc=f"Processing {label}"))
        
        # 결과 집계
        for segments_count in results:
            if segments_count == -1:
                stats["skipped_files"] += 1
            elif segments_count > 0:
                stats["processed_files"] += 1
                stats["extracted_segments"] += segments_count
            else:
                stats["failed_files"] += 1
    
    print(f"Successfully processed: {stats['processed_files']} files")
    print(f"Failed: {stats['failed_files']} files")
    print(f"Skipped (already processed): {stats['skipped_files']} files")
    print(f"Total segments: {stats['extracted_segments']}")
    print(f"Average segments per file: {stats['extracted_segments'] / max(1, stats['processed_files']):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract segments from audio files recursively")
    parser.add_argument("--input", type=str, default="/data/datasets/real_musics",
                        help="Input directory with audio files")
    parser.add_argument("--output", type=str, default="/data/datasets/ai_detection_dataset_segment",
                        help="Output directory for segments")
    parser.add_argument("--workers", type=int, default=14,
                        help="Number of parallel workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for beat extraction")
    parser.add_argument("--max-duration", type=int, default=300,
                        help="Maximum audio duration in seconds")
    parser.add_argument("--min-duration", type=int, default=30,
                        help="Minimum audio duration in seconds")
    parser.add_argument("--labels", nargs='+', default=None,
                        help="Labels to process")
    
    args = parser.parse_args()
    
    segment_dataset(
        base_dir=args.input,
        output_base_dir=args.output,
        num_workers=args.workers,
        device=args.device,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        labels=args.labels
    )