import os
import time
import argparse
from pathlib import Path
import random

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


# ============================================================
# Utils
# ============================================================

def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum((2, 3))
    union = prob.sum((2, 3)) + target.sum((2, 3))
    dice = 1 - (2 * inter + 1) / (union + 1)
    return bce + dice.mean()


# ============================================================
# Dataset
# ============================================================

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

class SemanticSegDataset(Dataset):
    def __init__(self, data_root, split="train", img_size=640):
        self.img_dir = Path(data_root) / "images" / split
        self.mask_dir = Path(data_root) / "masks" / split
        assert self.img_dir.exists()
        assert self.mask_dir.exists()

        self.img_size = img_size
        self.samples = []

        for img_path in self.img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            stem = img_path.stem
            mask_path = self.mask_dir / f"{stem}.png"
            if mask_path.exists():
                self.samples.append({
                    "img": img_path,
                    "mask": mask_path,
                    "name": stem
                })

        print(f"[{split}] samples = {len(self.samples)} (img_size={self.img_size})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p = self.samples[idx]

        bgr = cv2.imread(str(p["img"]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        mask_img = cv2.imread(str(p["mask"]), cv2.IMREAD_GRAYSCALE)

        rgb = cv2.resize(rgb, (self.img_size, self.img_size))
        mask = cv2.resize(mask_img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

        rgb = rgb.astype(np.float32) / 255.0
        rgb = rgb.transpose(2, 0, 1)

        return {
            "image": torch.from_numpy(rgb),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "name": p["name"],
        }


# ============================================================
# Model
# ============================================================

def build_deeplabv3plus(encoder_name="resnet50",
                        encoder_weights="imagenet",
                        in_channels=3,
                        out_channels=1):
    """
    encoder_name 可选：
      resnet50, resnet101
      efficientnet-b3, efficientnet-b5
      mit_b0, mit_b1, mit_b3, mit_b5（SMP 全支持）
    """
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    model.eval()
    tot_loss, tot_iou, cnt = 0.0, 0.0, 0

    for batch in loader:
        img = batch["image"].to(device)
        gt = batch["mask"].to(device)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(img)
            loss = bce_dice_loss(logits, gt)

        tot_loss += loss.item()

        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()

        B = img.shape[0]
        for i in range(B):
            pm = pred[i, 0].cpu().numpy().astype(bool)
            gm = gt[i, 0].cpu().numpy().astype(bool)

            inter = (pm & gm).sum()
            union = (pm | gm).sum()
            iou = inter / (union + 1e-6)
            tot_iou += iou
            cnt += 1

    return tot_loss / len(loader), tot_iou / cnt


# ============================================================
# Train Loop
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    set_seed(args.seed)

    # 权重输出目录
    weights_dir = Path(args.weights_root) / "deeplabv3plus"
    ensure_dir(weights_dir)
    print("\n权重保存目录:", weights_dir)

    # Dataset + Dataloader
    train_ds = SemanticSegDataset(args.data_root, "train", img_size=args.img_size)
    val_ds = SemanticSegDataset(args.data_root, "valid", img_size=args.img_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Model
    model = build_deeplabv3plus(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        out_channels=1
    ).to(device)

    print("Backbone:", args.encoder)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_miou = 0.0

    # Train loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"[DeepLabV3+] Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            img = batch["image"].to(device)
            gt = batch["mask"].to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(img)
                loss = bce_dice_loss(logits, gt)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = total_loss / len(train_loader)

        # Validation
        val_loss, miou = evaluate(model, val_loader, device, amp=args.amp)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | mIoU={miou:.4f} | "
              f"Time={time.time()-t0:.1f}s")

        # Save latest
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "miou": miou,
        }
        torch.save(ckpt, weights_dir / f"epoch{epoch+1:03d}.pth")

        # Save best
        if miou > best_miou:
            best_miou = miou
            ckpt["best_miou"] = best_miou
            torch.save(ckpt, weights_dir / "best.pth")
            print(f"🔥 New Best mIoU={best_miou:.4f} -> 已保存 best.pth")

    print("\n训练完成！最佳模型:", weights_dir / "best.pth")


# ============================================================
# Main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser("Train DeepLabV3+ (SMP)")

    ap.add_argument("--data_root", default="/root/autodl-tmp/dataset_seg")
    ap.add_argument("--weights_root", default="/root/autodl-tmp/Unet/unet_weights")

    ap.add_argument("--encoder", type=str, default="resnet50",
                    help="resnet50 / resnet101 / mit_b3 / efficientnet-b3 ...")
    ap.add_argument("--encoder_weights", type=str, default="imagenet",
                    help="imagenet 或 None")

    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--val_batch_size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
