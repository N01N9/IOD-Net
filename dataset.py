import torch
import torch.nn.functional as F
import random
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, Audio
from huggingface_hub import list_repo_files  # 추가된 라이브러리

class NCGMStreamingDataset(IterableDataset):
    def __init__(self, repo_id, split="train", sample_rate=16000, duration=20.0, mask_prob=0.5, skip_count=0):
        # ... (샤드 로직 기존 유지) ...
        print(f"🔍 {repo_id}에서 샤드 목록을 조회 중...")
        all_files = list_repo_files(repo_id, repo_type="dataset")
        tar_shards = sorted([f for f in all_files if f.endswith(".tar") and split in f])
        
        shards_to_skip = skip_count // 1000
        remaining_offset = skip_count % 1000
        selected_shards = tar_shards[shards_to_skip:]
        
        self.audio_ds = load_dataset(
            repo_id, data_files=selected_shards, split=split, streaming=True
        ).cast_column("wav", Audio(sampling_rate=sample_rate))
        
        meta_stream = load_dataset(repo_id, data_files="metadata.jsonl", split="train", streaming=True)
        self.meta = {item["file_name"].replace(".wav", ""): item["utterances"] for item in meta_stream}
        
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * duration)
        self.num_frames = 2001
        self.max_speakers = 6
        
        # [수정 1] 마스킹 확률을 기본 0.5(50%)로 상향하여 무음 학습 기회 확대
        self.mask_prob = mask_prob

    def apply_aggressive_masking(self, audio, target_mask):
        """
        [수정 2] 더 길고 다양한 무음 구간을 생성하여 VAD 편향 제거
        """
        if random.random() < self.mask_prob:
            # 1~2초(50~250프레임)가 아닌 1~8초(100~800프레임)까지 대폭 늘림
            mask_len_frames = random.randint(100, 800) 
            start_frame = random.randint(0, self.num_frames - mask_len_frames - 1)
            end_frame = start_frame + mask_len_frames
            
            # 절대적인 0(Digital Zero) 대신 아주 미세한 노이즈를 섞어 현실적인 무음 구현
            # 이는 모델이 "완전한 0"이 아닌 "작은 소리"도 무음으로 보게 함
            audio[start_frame * 160 : end_frame * 160] = torch.randn(mask_len_frames * 160) * 0.0001
            
            # 해당 구간의 정답 마스크를 0으로 확실하게 밀어버림 (VAD 타겟이 0이 됨)
            target_mask[start_frame:end_frame, :] = 0.0
            
        return audio, target_mask

    def __iter__(self):
        for item in self.audio_ds:
            # Try multiple key formats
            raw_filename = item.get("file_name", "")
            key_candidates = [
                item.get("__key__", ""),
                raw_filename,
                raw_filename.replace(".wav", ""),
                raw_filename.split("/")[-1].replace(".wav", "")
            ]
            
            # Find the first key that exists in metadata
            key = None
            for k in key_candidates:
                if k and k in self.meta:
                    key = k
                    break
            
            if key is None:
                # print(f"Skipping: {key_candidates} not found in metadata keys (sample: {list(self.meta.keys())[:5]})") 
                continue

            # [수정 3] 5%의 확률로 '완전 무음' 샘플을 생성 (Hard Negative)
            # 모델이 아무 소리도 없을 때 VAD가 0이 나와야 함을 강제로 학습
            force_total_silence = random.random() < 0.05
            
            audio = torch.tensor(item["wav"]["array"], dtype=torch.float32)
            if audio.ndim > 1: audio = audio.mean(dim=-1)
            
            # 패딩 및 커팅
            if audio.numel() > self.max_samples:
                audio = audio[:self.max_samples]
            else:
                audio = F.pad(audio, (0, self.max_samples - audio.numel()))
            
            target_mask = torch.zeros(self.num_frames, self.max_speakers)
            exist_target = torch.zeros(self.max_speakers)
            
            if not force_total_silence:
                utterances = self.meta[key]
                speakers = sorted(set(u["speaker"] for u in utterances))
                
                for i, spk in enumerate(speakers[:self.max_speakers]):
                    exist_target[i] = 1.0
                    for u in utterances:
                        if u["speaker"] == spk:
                            s, e = int(u["start"] * 100), int(u["end"] * 100)
                            target_mask[max(0, s):min(self.num_frames, e), i] = 1.0
                
                # 공격적인 마스킹 적용
                audio, target_mask = self.apply_aggressive_masking(audio, target_mask)
            else:
                # 완전 무음인 경우 오디오와 마스크 모두 0 (미세 노이즈 추가)
                audio = torch.randn_like(audio) * 0.0001
                # target_mask와 exist_target은 이미 0으로 초기화됨
            
            yield {
                "audio": audio, 
                "t": torch.linspace(0, 1, steps=self.num_frames).unsqueeze(-1), 
                "target_mask": target_mask, 
                "exist_target": exist_target
            }

# get_dataloader 등 나머지 인터페이스는 동일하게 유지하되 mask_prob=0.5 권장

def get_dataloader(repo_id, batch_size=8, mask_prob=0.5, skip_count=0, num_workers=0):
    """기존 호출 인터페이스 유지"""
    dataset = NCGMStreamingDataset(
        repo_id=repo_id, 
        mask_prob=mask_prob, 
        skip_count=skip_count
    )
    
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)