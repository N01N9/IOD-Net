import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment  # [추가] 헝가리안 알고리즘

# ==========================================
# 1. Conv-TasNet 스타일의 TCN 블록 (유지)
# ==========================================
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, out_channels)
        
        self.d_conv = nn.Conv1d(
            out_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, 
            groups=out_channels, bias=True
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, out_channels)
        
        self.res_out = nn.Conv1d(out_channels, in_channels, 1)
        self.skip_out = nn.Conv1d(out_channels, in_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.norm1(self.prelu1(self.conv1x1(x)))
        out = self.norm2(self.prelu2(self.d_conv(out)))
        out = self.dropout(out)
        res = self.res_out(out)
        skip = self.skip_out(out)
        return residual + res, skip

class TCNReconstructor(nn.Module):
    """
    [이전 수정 유지] input_dim과 output_dim을 분리하여 차원 충돌 해결
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256, kernel_size=3, num_blocks=4):
        super().__init__()
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            self.blocks.append(
                TemporalBlock(hidden_dim, hidden_dim, kernel_size, stride=1, 
                              dilation=dilation, padding=padding)
            )
        self.output_conv = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(hidden_dim, output_dim, 1), # 출력 차원 맞춤
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.input_conv(x)
        skip_connection = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_connection = skip_connection + skip
        return self.output_conv(skip_connection)

# ==========================================
# 2. TCN 기반 신호 제거기 (유지)
# ==========================================
class LinearReconstructionEraser(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.epsilon = 1e-6
        
        # [이전 수정 유지] 입출력 차원 분리 전달
        self.reconstructor = TCNReconstructor(
            input_dim=input_dim + 1,   # 입력: Feature(128) + Mask(1) = 129
            output_dim=input_dim,      # 출력: Mask for Feature(128) = 128
            hidden_dim=hidden_dim, 
            num_blocks=4 
        )

    def forward(self, features, probs):
        linear_features = torch.exp(features)
        x_in = torch.cat([linear_features, probs], dim=2) 
        x_in = x_in.transpose(1, 2) 
        
        ratio_mask = self.reconstructor(x_in)
        ratio_mask = ratio_mask.transpose(1, 2) # (B, T, 128)
        
        estimated_signal = linear_features * ratio_mask * probs
        linear_residual = linear_features - estimated_signal
        
        linear_residual = torch.relu(linear_residual) + self.epsilon
        return torch.log(linear_residual)

# ==========================================
# 3. 디코더 (Direct Mask Prediction) - 유지
# ==========================================
class DirectMaskDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + 1, hidden_dim)
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, features, prev_mask):
        x = torch.cat([features, prev_mask], dim=2)
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        logits = self.net(x)
        logits = logits.transpose(1, 2)
        return torch.sigmoid(logits)

# ==========================================
# 4. 인코더 (6-layer E-Branchformer) - 유지
# ==========================================
class EBranchformerLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        
        self.cgmlp_norm = nn.LayerNorm(d_model)
        # [이전 수정 유지] Linear -> Conv1d 교체 완료됨
        self.cgmlp = nn.Sequential(
            nn.Conv1d(d_model, d_ffn, kernel_size=1), 
            nn.GELU(),
            nn.Conv1d(d_ffn, d_ffn, kernel_size=31, stride=1, padding=15, groups=d_ffn),
            nn.GELU(),
            nn.Conv1d(d_ffn, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.merge_proj = nn.Linear(d_model * 2, d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x_norm = self.attn_norm(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        
        x_local = self.cgmlp_norm(x)
        x_local = x_local.transpose(1, 2)
        x_local = self.cgmlp(x_local)
        x_local = x_local.transpose(1, 2)
        
        concat_feat = torch.cat([x_attn, x_local], dim=-1)
        merged = self.merge_proj(concat_feat)
        return self.final_norm(residual + self.dropout(merged))

class EBranchformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=256, num_layers=6):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.layers = nn.ModuleList([
            EBranchformerLayer(d_model=output_dim, nhead=4, d_ffn=1024)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return x

# ==========================================
# 5. 전체 모델 (Iterative Diarizer)
# ==========================================
class IterativeDiarizer(nn.Module):
    def __init__(self, input_feat_dim=128, d_model=128, max_speakers=6):
        super().__init__()
        self.encoder = EBranchformerEncoder(input_feat_dim, d_model, num_layers=6)
        self.decoder = DirectMaskDecoder(d_model, hidden_dim=d_model)
        self.projector = LinearReconstructionEraser(input_dim=d_model, hidden_dim=d_model)
        self.max_speakers = max_speakers

    def forward(self, audio_features, vad_mask=None):
        batch_size, time, _ = audio_features.shape
        encoded_features = self.encoder(audio_features)
        
        if vad_mask is None:
            current_mask = torch.zeros(batch_size, time, 1).to(audio_features.device)
        else:
            current_mask = vad_mask
            
        residual_features = encoded_features.clone()
        all_speaker_probs = []

        for i in range(self.max_speakers):
            speaker_prob = self.decoder(residual_features, current_mask)
            all_speaker_probs.append(speaker_prob)
            residual_features = self.projector(encoded_features, speaker_prob)
            current_mask = torch.max(current_mask, speaker_prob)
            
        return torch.cat(all_speaker_probs, dim=2)

# ==========================================
# 6. [NEW] 헝가리안 알고리즘 기반 PIT Loss
# ==========================================
class HungarianPITLoss(nn.Module):
    """
    Max Speaker가 많을 때(예: 6명 이상) 순열(Factorial) 대신
    헝가리안 알고리즘(O(n^3))을 사용하여 최적의 화자 매칭을 수행하는 Loss.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss(reduction='none')

    def forward(self, predictions, targets):
        """
        predictions: (Batch, Time, Spk)
        targets: (Batch, Time, Spk)
        """
        b, t, s = predictions.shape
        
        # 1. 모든 쌍에 대한 비용 행렬(Cost Matrix) 계산
        # Broadcasting을 위해 차원 확장
        # Pred: (B, T, S, 1), Target: (B, T, 1, S)
        # 결과: (B, T, S, S) -> 각 (Pred_i, Target_j) 조합의 Loss
        p_exp = predictions.unsqueeze(3)
        t_exp = targets.unsqueeze(2)
        
        # 전체 Loss 맵 계산 (Gradient Flow 유지를 위해 여기서 수행)
        # pairwise_loss: (B, S, S) -> Time 축 평균
        pairwise_loss = F.binary_cross_entropy(
            p_exp.expand(-1, -1, -1, s), 
            t_exp.expand(-1, -1, s, -1), 
            reduction='none'
        ).mean(dim=1)
        
        total_loss = 0.0
        
        # 2. 배치별로 헝가리안 매칭 수행
        # 매칭 과정(Indices 찾기)은 Gradient가 필요 없으므로 detach().numpy()
        cost_matrices = pairwise_loss.detach().cpu().numpy()
        
        for i in range(b):
            cost_mat = cost_matrices[i] # (S, S)
            
            # Scipy로 최적의 할당 인덱스 찾기 (row=pred_idx, col=target_idx)
            row_idx, col_idx = linear_sum_assignment(cost_mat)
            
            # 3. 찾은 인덱스로 실제 Loss 값 추출 및 합산
            # pairwise_loss[i][row, col] 값들만 골라서 평균냄
            total_loss += pairwise_loss[i, row_idx, col_idx].mean()
            
        return total_loss / b

# ==========================================
# 실행부 (Main)
# ==========================================
if __name__ == "__main__":
    # 설정
    BATCH_SIZE = 4
    TIME_STEPS = 2000
    FEAT_DIM = 128
    MAX_SPEAKERS = 6  # 6명 설정
    
    model = IterativeDiarizer(FEAT_DIM, d_model=128, max_speakers=MAX_SPEAKERS)
    
    # [변경] HungarianPITLoss 사용
    loss_fn = HungarianPITLoss()
    
    print(f">>> 모델 초기화 완료 (Max Speakers: {MAX_SPEAKERS})")
    print(">>> 적용된 모듈: 6-Layer E-Branchformer, TCN Eraser, Hungarian PIT Loss")

    inputs = torch.randn(BATCH_SIZE, TIME_STEPS, FEAT_DIM)
    targets = torch.randint(0, 2, (BATCH_SIZE, TIME_STEPS, MAX_SPEAKERS)).float()
    
    # Forward
    predictions = model(inputs)
    print(f"Prediction Shape: {predictions.shape}")
    
    # Loss Calc
    loss = loss_fn(predictions, targets)
    print(f"Hungarian PIT Loss: {loss.item():.4f}")