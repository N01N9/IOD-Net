import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import argparse
import os
from tqdm import tqdm

from model import IterativeDiarizer, SortedBCELoss
from dataset import get_dataloader

# ==========================================
# Configuration
# ==========================================
# User should replace this with their actual Hugging Face Dataset Repo ID
# Example: "organization/my-speech-dataset"
DEFAULT_REPO_ID = "N02N9/ncgm-voxceleb" 

def get_args():
    parser = argparse.ArgumentParser(description="Train IOD-Net")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID, 
                        help="Hugging Face Dataset Repo ID (required)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--feat_dim", type=int, default=128, help="Feature dimension (Mel bins)")
    parser.add_argument("--d_model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--max_speakers", type=int, default=6, help="Max speakers to separate")
    parser.add_argument("--duration", type=float, default=20.0, help="Audio duration in seconds")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log directory")
    return parser.parse_args()

class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=160, win_length=400):
        """
        Log Mel Spectrogram Extractor
        Default settings aim for 10ms hop size (160 samples @ 16kHz)
        """
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
        # return: (Batch, Time, n_mels)
        mel = self.mel_spec(waveform)
        log_mel = self.amplitude_to_db(mel)
        return log_mel.transpose(1, 2)

def train(args):
    # TensorBoard Writer
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=args.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running training on: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Dataset & DataLoader
    print(f"Initializing DataLoader from repo: {args.repo_id}")
    try:
        dataloader = get_dataloader(
            repo_id=args.repo_id, 
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"\n[Error] Failed to load dataset from '{args.repo_id}'.")
        print(f"Please specify a valid Hugging Face dataset ID via --repo_id argument or edit train.py.\nDetail: {e}")
        return

    # 2. Model & Preprocessing
    feature_extractor = LogMelSpectrogram(n_mels=args.feat_dim).to(device)
    
    model = IterativeDiarizer(
        input_feat_dim=args.feat_dim, 
        d_model=args.d_model, 
        max_speakers=args.max_speakers
    ).to(device)
    
    loss_fn = SortedBCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # --- Resume Logic ---
    start_epoch = 0
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"Loading checkpoint from '{args.resume_from}'")
            checkpoint = torch.load(args.resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # [중요] 체크포인트의 LR 대신 현재 설정한 LR을 강제 적용
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            print(f"Optimizer LR updated to: {args.lr}")
            
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint not found at '{args.resume_from}'. Starting from scratch.")

    # 3. Training Loop
    model.train()
    global_step = 0
    
    try:
        for epoch in range(start_epoch, args.epochs):
            total_loss = 0.0
            batch_count = 0
            
            # NOTE: NCGMStreamingDataset is iterable, so we iterate until it yields all shards or we break
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for batch in progress_bar:
                audio = batch['audio'].to(device)       # (B, Samples)
                targets = batch['target_mask'].to(device) # (B, Time, Spk)
                
                # Feature Extraction
                with torch.no_grad():
                    features = feature_extractor(audio) # (B, F_Time, D)
                
                # Align Time Dimension
                # MelSpectrogram time steps might slightly differ from target_mask (2001)
                # We crop to the minimum length to math
                min_len = min(features.shape[1], targets.shape[1])
                features = features[:, :min_len, :]
                targets = targets[:, :min_len, :]
                
                # Forward Pass
                # IterativeDiarizer recursively finds speakers
                predictions = model(features) # (B, Time, Max_Speakers)
                
                # Align Output if needed (DirectMaskDecoder preserves time usually)
                if predictions.shape[1] != targets.shape[1]:
                    min_len_out = min(predictions.shape[1], targets.shape[1])
                    predictions = predictions[:, :min_len_out, :]
                    targets = targets[:, :min_len_out, :]

                # Compute Loss
                loss = loss_fn(predictions, targets)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                global_step += 1
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # TensorBoard Logging (Batch Level)
                if global_step % 10 == 0:
                    writer.add_scalar("Loss/train_step", loss.item(), global_step)

            # End of Epoch
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
                
                # TensorBoard Logging (Epoch Level)
                writer.add_scalar("Loss/train_epoch", avg_loss, epoch + 1)

                # Save Checkpoint
                ckpt_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, ckpt_path)
                print(f"Checkpoint saved to {ckpt_path}")
            else:
                print("Warning: No batches processed. Check dataset configuration.")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")
        ckpt_path = os.path.join(args.checkpoint_dir, "model_interrupted.pt")
        torch.save({
            'epoch': epoch, # Save current epoch
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / max(1, batch_count),
        }, ckpt_path)
        print(f"Emergency checkpoint saved to {ckpt_path}")
        writer.close()
        
    writer.close()

if __name__ == "__main__":
    args = get_args()
    train(args)
