import torch
import torch.nn as nn
import torchaudio
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from model import IterativeDiarizer

# Train.py와 동일한 전처리 클래스
class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=160, win_length=400):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20,
            f_max=8000,
            normalized=True
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform):
        # waveform: (Batch, Samples)
        mel = self.mel_spec(waveform)
        log_mel = self.amplitude_to_db(mel)
        return log_mel.transpose(1, 2)

def rttm_format(file_id, spk_id, start, duration):
    """RTTM 포맷 문자열 생성"""
    return f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {spk_id} <NA> <NA>"

def merge_intervals(intervals, gap_threshold=0.5):
    """짧은 공백은 무시하고 구간 합치기"""
    if not intervals:
        return []
    
    merged = []
    curr_start, curr_end = intervals[0]
    
    for next_start, next_end in intervals[1:]:
        if next_start - curr_end < gap_threshold:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
            
    merged.append((curr_start, curr_end))
    return merged

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 1. 모델 로드
    # 학습 시 설정했던 파라미터와 동일해야 합니다.
    print(f"Loading model from {args.checkpoint}...")
    model = IterativeDiarizer(
        input_feat_dim=args.feat_dim, 
        d_model=args.d_model, 
        max_speakers=args.max_speakers
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")

    # 2. 오디오 로드 및 전처리
    print(f"Processing audio: {args.input_audio}")
    waveform, sample_rate = torchaudio.load(args.input_audio)
    
    # 리샘플링 (16kHz 필수)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # 모노로 변환
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    feature_extractor = LogMelSpectrogram(n_mels=args.feat_dim).to(device)
    
    # GPU로 이동 및 배치 차원 추가
    waveform = waveform.to(device)
    
    with torch.no_grad():
        features = feature_extractor(waveform) # (1, Time, Feat_Dim)
        
        # 3. 추론
        # (1, Time, Max_Speakers) -> sigmoid 적용된 확률값
        predictions = model(features) 
        
    # 4. 결과 해석
    probs = predictions.squeeze(0).cpu().numpy() # (Time, Spk)
    num_frames = probs.shape[0]
    frame_duration = 0.01 # 10ms (hop_length 160 / sr 16000)
    
    # 시각화 (이미지 저장)
    if args.output_image:
        print(f"Saving visualization to {args.output_image}...")
        plt.figure(figsize=(20, 6))
        
        # (Time, Spk) -> (Spk, Time)으로 전치하여 히트맵 그리기
        # aspect='auto'로 설정하여 시간축이 길어도 찌그러지지 않게 함
        plt.imshow(probs.T, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
        
        plt.colorbar(label='Probability')
        plt.title(f"Speaker Diarization Heatmap: {os.path.basename(args.input_audio)}")
        plt.xlabel("Time (Frame)")
        plt.ylabel("Speaker Index")
        plt.yticks(range(args.max_speakers))
        
        plt.tight_layout()
        plt.savefig(args.output_image)
        plt.close()
    
    # 임계값 적용
    binary_mask = probs > args.threshold
    
    # 결과 출력 및 RTTM 저장
    file_id = os.path.splitext(os.path.basename(args.input_audio))[0]
    output_path = args.output_rttm if args.output_rttm else f"{file_id}.rttm"
    
    print("\n[Diarization Result]")
    with open(output_path, 'w') as f:
        for spk_idx in range(args.max_speakers):
            spk_activity = binary_mask[:, spk_idx]
            
            # 활성 구간 찾기
            intervals = []
            is_active = False
            start_frame = 0
            
            for t, active in enumerate(spk_activity):
                if active and not is_active:
                    is_active = True
                    start_frame = t
                elif not active and is_active:
                    is_active = False
                    intervals.append((start_frame * frame_duration, t * frame_duration))
            
            if is_active:
                intervals.append((start_frame * frame_duration, num_frames * frame_duration))
            
            # 구간 병합 및 출력
            merged_intervals = merge_intervals(intervals)
            
            if merged_intervals:
                print(f"Speaker {spk_idx}:")
                for start, end in merged_intervals:
                    duration = end - start
                    line = rttm_format(file_id, f"spk_{spk_idx}", start, duration)
                    f.write(line + "\n")
                    print(f"  - {start:.2f}s ~ {end:.2f}s")
            else:
                pass # 이 화자는 말하지 않음

    print(f"\nSaved RTTM file to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference IOD-Net")
    parser.add_argument("--input_audio", type=str, required=True, help="Path to input wav file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output_rttm", type=str, default=None, help="Path to output RTTM file")
    parser.add_argument("--output_image", type=str, default=None, help="Path to output visualization image (.png)")
    
    # Model Config (Must match training)
    parser.add_argument("--feat_dim", type=int, default=128, help="Feature dimension")
    parser.add_argument("--d_model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--max_speakers", type=int, default=6, help="Max speakers")
    parser.add_argument("--threshold", type=float, default=0.5, help="Voice activity threshold")
    
    args = parser.parse_args()
    inference(args)
