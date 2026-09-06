"""
PyTorch GPU U-Net Trainer for Synthetic RFI Spectrogram Dataset
===============================================================
Natively utilizes CUDA hardware acceleration on your NVIDIA GeForce RTX 3060 GPU.
Implements the 2D U-Net architecture (Akeret et al. 2017) with GPU acceleration,
automatic checkpoint saving/restoration, and paper metric evaluations (ROC AUC, PR AUC, F1).
"""

import os
import glob
import json
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. PyTorch Dataset for Synthetic RFI .npy pairs
# ---------------------------------------------------------------------------
class RFINpyDataset(Dataset):
    """
    Loads synthetic RFI spectrograms and binary ground-truth masks.
    """
    def __init__(self, images_dir, masks_dir):
        self.image_files = sorted(glob.glob(os.path.join(images_dir, "*.npy")))
        self.mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.npy")))
        assert len(self.image_files) == len(self.mask_files) and len(self.image_files) > 0, (
            f"No matching image/mask .npy pairs found in {images_dir} / {masks_dir}"
        )
        print(f"RFINpyDataset: Loaded {len(self.image_files)} pairs from {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load raw numpy arrays
        img = np.load(self.image_files[idx]).astype(np.float32)  # shape (H, W)
        mask = np.load(self.mask_files[idx]).astype(np.int64)   # shape (H, W), 0=clean, 1=RFI

        # Per-image channel normalization
        mean, std = img.mean(), img.std() + 1e-6
        img_norm = (img - mean) / std

        # Convert to PyTorch Tensors
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(mask)                  # (H, W)

        return img_tensor, mask_tensor, self.image_files[idx]


# ---------------------------------------------------------------------------
# 2. PyTorch U-Net Architecture (Akeret et al. 2017)
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetRFI(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, features_root=64, layers=3):
        super().__init__()
        self.layers = layers
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (Downsampling path)
        feats = features_root
        curr_in = in_channels
        for _ in range(layers):
            self.downs.append(DoubleConv(curr_in, feats))
            curr_in = feats
            feats *= 2

        # Bottleneck
        self.bottleneck = DoubleConv(curr_in, feats)

        # Decoder (Upsampling path)
        for _ in range(layers):
            self.ups.append(nn.ConvTranspose2d(feats, feats // 2, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feats, feats // 2))
            feats //= 2

        # Final Classifier 1x1 Conv
        self.final_conv = nn.Conv2d(features_root, n_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsample
        for i in range(self.layers):
            x = self.downs[i](x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsample
        skip_connections = skip_connections[::-1]
        for i in range(self.layers):
            x = self.ups[i * 2](x)
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i * 2 + 1](x)

        return self.final_conv(x)


# ---------------------------------------------------------------------------
# 3. Training Loop with CUDA GPU Acceleration & Checkpointing
# ---------------------------------------------------------------------------
def train(args, device):
    print(f"\n{'='*70}")
    print(f" Starting PyTorch U-Net Training on GPU: {torch.cuda.get_device_name(device)}")
    print(f"{'='*70}\n")

    # Dataset paths
    train_dir = os.path.join(args.dataset_dir, "train")
    val_dir = os.path.join(args.dataset_dir, "val")
    if not os.path.exists(os.path.join(val_dir, "images")):
        val_dir = os.path.join(args.dataset_dir, "test")

    train_dataset = RFINpyDataset(
        os.path.join(train_dir, "images"),
        os.path.join(train_dir, "masks")
    )
    val_dataset = RFINpyDataset(
        os.path.join(val_dir, "images"),
        os.path.join(val_dir, "masks")
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize Network
    model = UNetRFI(in_channels=1, n_classes=2, features_root=args.features_root, layers=args.layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.2, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints_gpu")
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest_model.pt")

    start_epoch = 0
    # Restore checkpoint if requested and available
    if args.restore and os.path.exists(latest_ckpt_path):
        print(f"Resuming training from GPU checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Successfully restored weights! Resuming from Epoch {start_epoch}")
    else:
        print("Starting training from fresh weights.")

    # Mixed Precision Scaler for RTX 3060 Tensor Cores speedup
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_fp16)

    # Main Training Loop
    total_start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for step, (images, masks, _) in enumerate(train_loader):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=args.use_fp16):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (step + 1) % args.display_step == 0 or (step + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{args.epochs}] | Step [{step+1}/{len(train_loader)}] | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        print(f"--> Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, latest_ckpt_path)

    print(f"\nTraining Complete in {(time.time() - total_start_time)/60:.2f} minutes!")
    return model, latest_ckpt_path, val_loader


# ---------------------------------------------------------------------------
# 4. Evaluation: ROC / PR / F1 Metrics & Prediction Panel
# ---------------------------------------------------------------------------
def evaluate(model, checkpoint_path, val_loader, output_dir, device, n_eval=20):
    print(f"\nEvaluating trained model on GPU validation dataset...")
    model.eval()

    all_preds = []
    all_targets = []

    eval_count = 0
    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1, :, :].cpu().numpy()  # Probability of RFI class

            all_preds.append(probs.flatten())
            all_targets.append(masks.numpy().flatten())

            eval_count += images.size(0)
            if eval_count >= n_eval:
                break

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # Compute Metrics
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_f1 = float(np.max(f1_scores))

    eval_results = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "max_f1": best_f1,
        "n_eval_samples": int(eval_count)
    }

    eval_dir = os.path.join(output_dir, "eval_gpu")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    print("\n" + "="*50)
    print(" GPU Validation Results (Paper Metrics):")
    print(f"   ROC AUC : {roc_auc:.4f}  (paper: ~0.96)")
    print(f"   PR AUC  : {pr_auc:.4f}  (paper: ~0.92)")
    print(f"   Max F1  : {best_f1:.4f} (paper: ~0.85)")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="PyTorch GPU U-Net Trainer for Synthetic RFI Dataset")
    parser.add_argument("--dataset_dir", default="./Synthetic Dataset", help="Root of dataset containing train/ and test/")
    parser.add_argument("--output_dir", default="./unet_run_gpu", help="Output directory for checkpoints")
    parser.add_argument("--layers", type=int, default=3, help="U-Net depth")
    parser.add_argument("--features_root", type=int, default=64, help="Features root (default: 64)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for GPU (default: 1)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--display_step", type=int, default=5, help="Print status every N steps")
    parser.add_argument("--restore", action="store_true", default=True, help="Restore from GPU checkpoint if available")
    parser.add_argument("--use_fp16", action="store_true", default=True, help="Use FP16 Automatic Mixed Precision on GPU")
    args = parser.parse_args()

    # Device verification
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible NVIDIA GPU not detected by PyTorch! Please check your GPU installation.")

    device = torch.device("cuda")
    print(f"Using GPU Device: {torch.cuda.get_device_name(device)}")

    model, ckpt_path, val_loader = train(args, device)
    evaluate(model, ckpt_path, val_loader, args.output_dir, device)


if __name__ == "__main__":
    main()
