import os
import time
import argparse
import random
from pathlib import Path
import sys

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 把 Awesome-U-Net 加到 sys.path
# ============================================================

# train_unet.py: /root/autodl-tmp/tools/train_unet.py
THIS_DIR = Path(__file__).resolve().parent          # /root/autodl-tmp/tools
PROJECT_ROOT = THIS_DIR.parent                      # /root/autodl-tmp
AWESOME_UNET_ROOT = PROJECT_ROOT / "Unet" / "Awesome-U-Net"

sys.path.insert(0, str(AWESOME_UNET_ROOT))
print("✅ Added to sys.path:", AWESOME_UNET_ROOT)

# 现在可以直接 import models.XXX 了
from models.unet import UNet
from models.attunet import AttU_Net
from models.unetpp import NestedUNet


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
    """
    logits: (B,1,H,W)
    target: (B,1,H,W) 0/1
    """
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    prob = torch.sigmoid(logits)
    smooth = 1.0
    inter = (prob * target).sum((2, 3))
    union = prob.sum((2, 3)) + target.sum((2, 3))
    dice = 1 - (2 * inter + smooth) / (union + smooth)
    return bce + dice.mean()


def compute_iou(pm, gm):
    """
    pm, gm: (H,W) 0/1
    """
    pm = pm.astype(bool)
    gm = gm.astype(bool)
    inter = (pm & gm).sum()
    uni = (pm | gm).sum()
    return float(inter) / float(uni + 1e-6)


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


# ============================================================
# Dataset: 与 SAM 微调使用同一套数据
# ============================================================

class SemanticSegDataset(Dataset):
    """
    只做语义分割 (前景/背景)，不再用 box prompt 和 SAM 的 ResizeLongestSide，
    直接 resize 到固定大小 (img_size, img_size)。

    ROOT/
      images/{split}/xxx.jpg
      masks/{split}/xxx.png  # 单通道 0/255
    """

    def __init__(self, data_root, split, img_size=256):
        self.img_dir = Path(data_root) / "images" / split
        self.mask_dir = Path(data_root) / "masks" / split
        assert self.img_dir.exists(), f"{self.img_dir} 不存在"
        assert self.mask_dir.exists(), f"{self.mask_dir} 不存在"

        self.img_size = img_size
        self.samples = []

        for img_path in self.img_dir.iterdir():
            if not img_path.is_file():
                continue
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

        print(f"[{split}] samples = {len(self.samples)} (UNet 语义分割)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p = self.samples[idx]

        # ---- 读图像 ----
        bgr = cv2.imread(str(p["img"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"读取图像失败: {p['img']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # ---- 读 mask ----
        mask_img = cv2.imread(str(p["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise RuntimeError(f"读取mask失败: {p['mask']}")

        # ---- resize 到统一大小 ----
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask_img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)  # 0/1

        # ---- 转成 tensor ----
        rgb = rgb.astype(np.float32) / 255.0  # [0,1]
        rgb = rgb.transpose(2, 0, 1)          # (3,H,W)

        img_t = torch.from_numpy(rgb)         # (3,H,W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)  # (1,H,W)

        return {
            "image": img_t,
            "mask": mask_t,
            "name": p["name"],
        }


# ============================================================
# 构建网络
# ============================================================

def build_model(model_class: str, in_channels: int = 3, out_channels: int = 1):
    """
    model_class: 字符串，例：
        "UNet" / "AttU_Net" / "NestedUNet"
    """

    name = model_class.lower()

    if name == "unet":
        # 优先用位置参数，适配大多数实现：(in_channels, out_channels) / (img_ch, output_ch) / (n_channels, n_classes)
        try:
            model = UNet(in_channels, out_channels)
        except TypeError:
            # 如果实现里根本不需要这两个参数（写死了 3->1），就直接无参构造
            model = UNet()

    elif name in ["attunet", "att_u_net", "attenunet"]:
        # Attention U-Net，一般也是 (img_ch, output_ch)
        try:
            model = AttU_Net(in_channels, out_channels)
        except TypeError:
            try:
                model = AttU_Net(img_ch=in_channels, output_ch=out_channels)
            except TypeError:
                model = AttU_Net()


    elif name in ["unet++", "unetpp", "nestedunet"]:

        # ✅ 对应你贴出来的 NestedUNet(num_classes, input_channels=3, deep_supervision=False)

        model = NestedUNet(

            num_classes=out_channels,  # 类别数：1（前景/背景）

            input_channels=in_channels,  # 输入通道：3

            deep_supervision=False

        )


    else:

        raise ValueError(f"未知的 model_class: {model_class} (支持: UNet / AttU_Net / UNet++)")

    return model

# ============================================================
# Evaluate
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    """
    在验证集上计算 Loss 和 mIoU
    """
    model.eval()
    tot_loss, tot_iou, cnt = 0.0, 0.0, 0

    for batch in loader:
        img = batch["image"].to(device)   # (B,3,H,W)
        gt = batch["mask"].to(device)     # (B,1,H,W)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(img)           # (B,1,H,W)
            loss = bce_dice_loss(logits, gt)

        tot_loss += loss.item()

        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()

        B = img.shape[0]
        for i in range(B):
            pm = pred[i, 0].detach().cpu().numpy()
            gm = gt[i, 0].detach().cpu().numpy()
            tot_iou += compute_iou(pm, gm)
            cnt += 1

    model.train()
    avg_loss = tot_loss / max(1, len(loader))
    miou = tot_iou / max(1, cnt)
    return avg_loss, miou


# ============================================================
# Train
# ============================================================

def train_unet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    set_seed(args.seed)

    # 权重根目录：unet_weights/{net_name}/
    weights_root = Path(args.weights_root)
    model_dir = weights_root / args.net.lower()
    ensure_dir(model_dir)
    print(f"权重保存目录: {model_dir}")

    # ===== Dataset =====
    train_ds = SemanticSegDataset(args.data_root, "train", img_size=args.img_size)
    val_ds = SemanticSegDataset(args.data_root, "valid", img_size=args.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # ===== Model =====
    model = build_model(args.net).to(device)
    print(model.__class__.__name__)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_miou = 0.0

    # ============================================================
    #            Train Loop
    # ============================================================
    for epoch in range(args.epochs):
        total_loss = 0.0
        t0 = time.time()
        model.train()

        pbar = tqdm(train_loader, desc=f"[{args.net}] Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            img = batch["image"].to(device)   # (B,3,H,W)
            gt = batch["mask"].to(device)     # (B,1,H,W)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(img)           # (B,1,H,W)
                loss = bce_dice_loss(logits, gt)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = total_loss / len(train_loader)

        # ====== Validation ======
        val_loss, miou = evaluate(
            model,
            val_loader,
            device,
            amp=args.amp
        )

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | mIoU={miou:.4f} | "
              f"Time={time.time()-t0:.1f}s")

        # 保存当前 epoch 权重
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "miou": miou,
        }
        torch.save(ckpt, str(model_dir / f"epoch{epoch+1:03d}.pth"))

        # 保存 best 权重
        if miou > best_miou:
            best_miou = miou
            ckpt["best_miou"] = best_miou
            torch.save(ckpt, str(model_dir / "best.pth"))
            print(f"🎉 New Best mIoU = {best_miou:.4f} -> 已保存到 {model_dir/'best.pth'}")

    print("训练完成！最优模型保存在:", model_dir / "best.pth")


# ============================================================
# Main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser("UNet Family Training for Semantic Segmentation (Foreground/Background)")

    ap.add_argument("--data_root", default="/root/autodl-tmp/dataset_seg",
                    help="与 SAM 微调时相同的数据根目录")
    ap.add_argument("--weights_root", default="/root/autodl-tmp/Unet/unet_weights",
                    help="所有 UNet 权重的根目录")
    ap.add_argument("--net", type=str, default="unet++",
                    help="选择网络结构: unet / attenunet / unet++")

    ap.add_argument("--img_size", type=int, default=256)
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
    print("=" * 60)
    print("UNet 家族前景/背景语义分割训练")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 60)
    train_unet(args)
