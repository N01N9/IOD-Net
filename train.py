import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import argparse
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# model.py에서 최신 변경된 클래스들 임포트
from model import IterativeDiarizer, HungarianPITLoss
from dataset import get_dataloader

# ==========================================
# Configuration
# ==========================================
DEFAULT_REPO_ID = "N02N9/ncgm-voxceleb" 

def get_args():
    parser = argparse.ArgumentParser(description="Train IOD-Net")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID, 
                        help="Hugging Face Dataset Repo ID")
    parser.add_argument("--batch_size", type=int, default=8, help="Physical Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--feat_dim", type=int, default=128, help="Feature dimension")
    parser.add_argument("--d_model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--max_speakers", type=int, default=6, help="Max speakers to separate") 
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps")
    return parser.parse_args()

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
        mel = self.mel_spec(waveform)
        log_mel = self.amplitude_to_db(mel)
        return log_mel.transpose(1, 2)

def train(args):
    writer = SummaryWriter(log_dir=args.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running training on: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Dataset & DataLoader
    print(f"Initializing DataLoader from repo: {args.repo_id}")
    dataloader = get_dataloader(
        repo_id=args.repo_id, 
        batch_size=args.batch_size
    )

    # 2. Model & Components
    feature_extractor = LogMelSpectrogram(n_mels=args.feat_dim).to(device)
    
    model = IterativeDiarizer(
        input_feat_dim=args.feat_dim, 
        d_model=args.d_model, 
        max_speakers=args.max_speakers
    ).to(device)
    
    loss_fn = HungarianPITLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume Logic
    start_epoch = 0
    if args.resume_from and os.path.isfile(args.resume_from):
        print(f"Loading checkpoint from '{args.resume_from}'")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # 3. Training Loop
    model.train()
    global_step = 0
    
    try:
        for epoch in range(start_epoch, args.epochs):
            total_loss = 0.0
            batch_count = 0
            
            optimizer.zero_grad() 
            
            # Streaming Dataset이라 len()이 없어 total을 모름 -> tqdm에 total 제거
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for i, batch in enumerate(progress_bar):
                audio = batch['audio'].to(device)
                targets = batch['target_mask'].to(device)
                
                with torch.no_grad():
                    features = feature_extractor(audio)
                    min_len = min(features.shape[1], targets.shape[1])
                    features = features[:, :min_len, :]
                    targets = targets[:, :min_len, :]
                
                predictions = model(features)
                
                if predictions.shape[1] != targets.shape[1]:
                    min_len_out = min(predictions.shape[1], targets.shape[1])
                    predictions = predictions[:, :min_len_out, :]
                    targets = targets[:, :min_len_out, :]

                loss = loss_fn(predictions, targets)
                
                # Gradient Accumulation
                loss = loss / args.accum_steps
                loss.backward()
                
                total_loss += loss.item() * args.accum_steps
                
                if (i + 1) % args.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # [수정됨] 스케줄러 스텝을 여기서 제거함 (에포크 단위로 이동)
                    
                    global_step += 1
                    
                    if global_step % 10 == 0:
                        writer.add_scalar("Loss/train_step", loss.item() * args.accum_steps, global_step)
                        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_step)
                
                progress_bar.set_postfix({"loss": f"{loss.item() * args.accum_steps:.4f}"})
                batch_count = i + 1 # 현재까지 처리한 배치 수 업데이트

            # End of Epoch
            # [수정됨] 스케줄러 업데이트를 에포크가 끝난 후 여기서 수행
            scheduler.step()
            
            if batch_count == 0: batch_count = 1
            avg_loss = total_loss / batch_count 
            
            print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
            writer.add_scalar("Loss/train_epoch", avg_loss, epoch + 1)

            ckpt_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        ckpt_path = os.path.join(args.checkpoint_dir, "model_interrupted.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss if 'avg_loss' in locals() else 0.0,
        }, ckpt_path)
        
    writer.close()

if __name__ == "__main__":
    args = get_args()
    train(args)