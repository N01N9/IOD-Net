import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import IterativeDiarizer, HungarianPITLoss
from train import LogMelSpectrogram, get_args  # Reuse components from train.py
from dataset import get_dataloader

def run_overfit_test():
    # Use arguments from train.py but override for overfitting test
    args = get_args()
    
    # Force settings for overfitting test
    args.batch_size = 4  # Larger batch to test full memory load and throughput
    args.epochs = 1000    # Ensure convergence
    
    print(">>> Starting Overfitting Test")
    print(f"Target Repo: {args.repo_id}")
    print("Goal: Loss should approach 0 (e.g., < 0.01)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Dataset - Get ONE single batch
    print("Loading one batch from dataset...")
    try:
        dataloader = get_dataloader(
            repo_id=args.repo_id, 
            batch_size=args.batch_size,
            num_workers=0  # Disable multiprocessing to prevent OOM (std::bad_alloc)
        )
        # Fetch single batch and Move to Device
        fixed_batch = next(iter(dataloader))
        
        fixed_audio = fixed_batch['audio'].to(device)
        fixed_targets = fixed_batch['target_mask'].to(device)
        
        print(f"Fixed Batch Shape - Audio: {fixed_audio.shape}, Targets: {fixed_targets.shape}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Model & Components
    # Increase n_fft to 1024 to cover 128 mels properly without empty filters
    feature_extractor = LogMelSpectrogram(n_mels=args.feat_dim, n_fft=1024, win_length=1024).to(device)
    
    model = IterativeDiarizer(
        input_feat_dim=args.feat_dim, 
        d_model=args.d_model, 
        max_speakers=args.max_speakers
    ).to(device)
    
    loss_fn = HungarianPITLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Slightly higher LR for quick convergence

    # Pre-calculate features once to purely test the Model/Decoder capacity (Optional, but faster)
    # But usually better to test full pipeline inclusive of Feature Extractor gradients if it learns (it doesn't here, it's just transform)
    with torch.no_grad():
        fixed_features_raw = feature_extractor(fixed_audio)
        # Time alignment
        min_len = min(fixed_features_raw.shape[1], fixed_targets.shape[1])
        fixed_features = fixed_features_raw[:, :min_len, :].detach() # Detach as we don't update feature extractor usually
        fixed_targets = fixed_targets[:, :min_len, :]

    print(f"Features prepared: {fixed_features.shape}")
    
    # 3. Training Loop on Fixed Batch
    model.train()
    
    losses = []
    
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Forward
        predictions = model(fixed_features)
        
        # Align Output (just in case)
        if predictions.shape[1] != fixed_targets.shape[1]:
            min_len_out = min(predictions.shape[1], fixed_targets.shape[1])
            predictions = predictions[:, :min_len_out, :]
            fixed_targets = fixed_targets[:, :min_len_out, :]
            
        # Loss
        loss = loss_fn(predictions, fixed_targets)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Logging
        current_loss = loss.item()
        losses.append(current_loss)
        pbar.set_description(f"Loss: {current_loss:.6f}")

        # Early stop if converged
        if current_loss < 0.001:
            print(f"\nExample converged early at epoch {epoch}!")
            break

    # 4. Result Analysis
    print("\n>>> Test Finished")
    print(f"Final Loss: {losses[-1]:.6f}")
    
    if losses[-1] < 0.05:
        print("✅ SUCCESS: Model successfully overfitted the batch.")
    else:
        print("❌ FAILURE: Model failed to overfit. Check model architecture or data labels.")

    # Visualize first sample in batch
    try:
        plot_results(losses, predictions, fixed_targets)
    except Exception as e:
        print(f"Skipping visualization: {e}")

def plot_results(losses, predictions, targets):
    """
    Saves a plot of Loss curve and Prediction vs Target for the first sample
    """
    plt.figure(figsize=(12, 6))
    
    # 1. Loss Curve
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title("Training Loss (Overfitting)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    # Select first sample from batch
    pred_sample = predictions[0].detach().cpu().numpy() # (Time, Spk)
    target_sample = targets[0].detach().cpu().numpy()   # (Time, Spk)
    
    # 2. Prediction Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(pred_sample.T, aspect='auto', origin='lower', vmin=0, vmax=1)
    plt.title("Prediction (Final Epoch)")
    plt.xlabel("Time")
    plt.ylabel("Speaker Channel")
    
    # 3. Target Heatmap
    plt.subplot(1, 3, 3)
    plt.imshow(target_sample.T, aspect='auto', origin='lower', vmin=0, vmax=1)
    plt.title("Target Ground Truth")
    plt.xlabel("Time")
    
    save_path = "overfit_result.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    run_overfit_test()
