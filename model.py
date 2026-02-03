import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 유사 화자 분리 핵심 모듈 (Orthogonal Projection)
# ==========================================
class SoftOrthogonalProjector(nn.Module):
    """
    [핵심] 찾아낸 화자(Prob)의 벡터 방향을 계산하고, 
    원본 특징에서 그 성분을 수학적으로 제거(Subtract)하여 
    유사 화자(Similar Speaker)의 미세한 특징만 남기는 모듈.
    """
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha  # 제거 강도 (1.0에 가까울수록 강력하게 제거)

    def forward(self, features, probs):
        # features: (Batch, Time, Dim)
        # probs: (Batch, Time, 1) -> 현재 찾은 화자의 확률맵
        
        # 1. 화자 대표 벡터(Centroid) 추출
        # 확률(probs)을 가중치로 사용하여 해당 화자가 활성화된 구간의 특징만 평균냅니다.
        denom = probs.sum(dim=1, keepdim=True) + 1e-6
        speaker_vector = (features * probs).sum(dim=1, keepdim=True) / denom # (B, 1, Dim)
        
        # 2. 벡터 정규화 (크기는 버리고 방향만 남김)
        speaker_vector = F.normalize(speaker_vector, p=2, dim=2)
        
        # 3. 투영(Projection) 계산
        # 현재 특징들(features)이 화자 벡터와 얼마나 닮았는지(내적) 계산
        dot_product = torch.matmul(features, speaker_vector.transpose(1, 2)) # (B, T, 1)
        
        # 투영된 성분 (즉, 화자 A라고 판단되는 성분 벡터)
        projection = dot_product * speaker_vector # (B, T, Dim)
        
        # 4. 잔차(Residual) 계산 (Soft Subtraction)
        # alpha * probs: 화자 A가 확실한 구간에서만 강하게 지웁니다.
        # 오버랩 구간 보존을 위해 Soft하게 뺍니다.
        suppression = probs * self.alpha
        residual_features = features - (projection * suppression)
        
        return residual_features

# ==========================================
# 2. 디코더 (Direct Mask Prediction)
# ==========================================
class DirectMaskDecoder(nn.Module):
    """
    Attractor나 임베딩 없이, 오디오 특징과 마스크 힌트만 보고 
    다음 화자를 바로 색칠(Segmentation)하는 U-Net 스타일 네트워크.
    """
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        # 입력: 오디오 특징(D) + 이전 마스크(1)
        self.input_proj = nn.Linear(input_dim + 1, hidden_dim)
        
        # 1D CNN (문맥 파악 + 국소 경계면 탐지)
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3), # 넓은 문맥
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1) # 최종 출력: 1채널 확률값
        )

    def forward(self, features, prev_mask):
        # features: (B, T, D)
        # prev_mask: (B, T, 1) -> "이 부분은 이미 찾았으니 무시해"라는 힌트
        
        # 1. 정보 결합 (Concatenation)
        x = torch.cat([features, prev_mask], dim=2)
        x = self.input_proj(x) # (B, T, Hidden)
        
        # 2. CNN 처리를 위해 차원 변경 (B, Hidden, T)
        x = x.transpose(1, 2)
        
        # 3. 마스크 예측
        logits = self.net(x)
        
        # 4. 차원 복구
        logits = logits.transpose(1, 2) # (B, T, 1)
        
        return torch.sigmoid(logits)

# ==========================================
# 3. 인코더 (Placeholder)
# ==========================================
class EncoderPlaceholder(nn.Module):
    """
    실전에서는 ESPnet 등의 'EBranchformerEncoder'를 사용해야 합니다.
    여기서는 코드 실행을 위해 Transformer로 모사했습니다.
    """
    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=4, dim_feedforward=1024, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        
    def forward(self, x):
        x = self.proj(x)
        return self.encoder(x)

# ==========================================
# 4. 전체 모델 (Main Model)
# ==========================================
class IterativeDiarizer(nn.Module):
    def __init__(self, input_feat_dim=128, d_model=128, max_speakers=3):
        super().__init__()
        
        # [The Ear] 고해상도 특징 추출
        self.encoder = EncoderPlaceholder(input_feat_dim, d_model)
        
        # [The Brain] 화자 분리기
        self.decoder = DirectMaskDecoder(d_model, hidden_dim=d_model)
        
        # [The Eraser] 유사 화자 제거기
        self.projector = SoftOrthogonalProjector(alpha=0.9)
        
        self.max_speakers = max_speakers

    def forward(self, audio_features, vad_mask=None):
        """
        Args:
            audio_features: (Batch, Time, Feat_Dim)
            vad_mask: (Batch, Time, 1) - 묵음=1, 소리=0 (없으면 자동 처리)
        """
        batch_size, time, _ = audio_features.shape
        
        # 1. 인코딩 (한 번만 수행)
        encoded_features = self.encoder(audio_features)
        
        # 2. 초기 마스크 설정 (VAD가 없으면 0으로 시작)
        if vad_mask is None:
            current_mask = torch.zeros(batch_size, time, 1).to(audio_features.device)
        else:
            current_mask = vad_mask
            
        # 잔차 특징 초기화 (처음엔 원본 그대로)
        residual_features = encoded_features.clone()
        
        all_speaker_probs = []

        # 3. 반복 루프 (Recursive Loop)
        for i in range(self.max_speakers):
            
            # A. 디코더: 현재 잔차 특징과 마스크를 보고 다음 화자 찾기
            # Loop 1: 가장 지배적인 화자(A) 찾기
            # Loop 2: A가 제거된 특징에서 다음 화자(B) 찾기 ...
            speaker_prob = self.decoder(residual_features, current_mask)
            all_speaker_probs.append(speaker_prob)
            
            # B. 직교 투영: 방금 찾은 화자 성분을 수학적으로 제거
            # (유사 화자 구별을 위해 방향성을 이용해 찢어냄)
            residual_features = self.projector(encoded_features, speaker_prob)
            
            # C. 마스크 업데이트: 찾은 화자도 "처리됨"으로 마킹
            # max()를 써서 기존 마스크를 유지하며 누적
            current_mask = torch.max(current_mask, speaker_prob)
            
        # 결과 결합: (Batch, Time, Max_Speakers) -> 채널 0이 첫 화자, 채널 1이 두 번째 화자...
        return torch.cat(all_speaker_probs, dim=2)

# ==========================================
# 5. Loss 함수 (Sortformer Style)
# ==========================================
class SortedBCELoss(nn.Module):
    """
    PIT(순열 계산) 대신, 정답지(Target)를 '발화 시작 시간' 순서로 정렬하여
    모델의 순차적 출력과 1:1 매칭시키는 Loss 함수.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss(reduction='none')

    def sort_targets(self, targets):
        batch_size, time, max_speakers = targets.shape
        sorted_targets = torch.zeros_like(targets)
        
        for b in range(batch_size):
            sample = targets[b].transpose(0, 1) # (Spk, Time)
            
            # 각 화자별 발화 시작 시간(Onset) 찾기
            onsets = torch.argmax(sample.float(), dim=1)
            activities = sample.sum(dim=1)
            
            # 말하지 않은 화자(Ghost)는 맨 뒤로 보내기
            onsets[activities == 0] = time + 999999 
            
            # 시작 시간 순으로 정렬 인덱스 생성
            sort_idx = torch.argsort(onsets)
            
            # 정답지 재배열
            sorted_targets[b] = targets[b, :, sort_idx]
            
        return sorted_targets

    def forward(self, predictions, targets):
        # 1. 정답지 정렬 (먼저 말한 사람이 채널 0으로 오도록)
        aligned_targets = self.sort_targets(targets)
        
        # 2. 채널 수 맞추기 (학습 시 보통 맞춰짐)
        if predictions.shape[2] != aligned_targets.shape[2]:
            min_ch = min(predictions.shape[2], aligned_targets.shape[2])
            predictions = predictions[:, :, :min_ch]
            aligned_targets = aligned_targets[:, :, :min_ch]

        # 3. Loss 계산 (1:1 매칭)
        loss = self.criterion(predictions, aligned_targets)
        return loss.mean()

# ==========================================
# 6. 실행 및 검증 (Training Step Simulation)
# ==========================================
if __name__ == "__main__":
    # 설정
    BATCH_SIZE = 4
    TIME_STEPS = 2000  # 20 seconds audio (assuming 10ms hop size)
    FEAT_DIM = 128
    MAX_SPEAKERS = 4
    
    # 모델 및 손실함수 생성
    model = IterativeDiarizer(FEAT_DIM, d_model=128, max_speakers=MAX_SPEAKERS)
    loss_fn = SortedBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(">>> 모델 초기화 완료")

    # 1. 더미 데이터 생성 (입력)
    inputs = torch.randn(BATCH_SIZE, TIME_STEPS, FEAT_DIM)
    
    # 2. 더미 정답지 생성 (랜덤하지만 순서가 뒤섞여 있다고 가정)
    # 실제로는 (Batch, Time, Real_Speakers) 형태
    targets = torch.randint(0, 2, (BATCH_SIZE, TIME_STEPS, MAX_SPEAKERS)).float()
    
    # --- 학습 단계 시뮬레이션 ---
    optimizer.zero_grad()
    
    # A. 순방향 전파 (Forward)
    # 모델은 순차적으로 화자를 찾아 (Batch, Time, 3)을 출력합니다.
    # Loop 1 -> 제일 먼저 말한 사람 (채널 0)
    # Loop 2 -> 그다음 사람 (채널 1) ...
    predictions = model(inputs)
    
    print(f"Prediction Shape: {predictions.shape}")
    
    # B. 손실 계산 (Loss)
    # 내부적으로 targets를 정렬하여 predictions 순서에 맞춥니다.
    loss = loss_fn(predictions, targets)
    
    print(f"Calculated Loss: {loss.item():.4f}")
    
    # C. 역전파 (Backward)
    loss.backward()
    optimizer.step()
    
    print(">>> 역전파 및 가중치 업데이트 성공!")
    print("\n[검증 포인트]")
    print("1. Encoder는 오디오 특징을 추출함")
    print("2. Decoder는 반복적으로 다음 화자를 찾음")
    print("3. Projector는 찾은 화자를 수학적으로 제거(유사 화자 분리)")
    print("4. SortedLoss는 정답지의 순서를 모델 출력 순서에 맞춰줌")