import os
import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ================== UNet 系列路径 ==================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
AWESOME_UNET_ROOT = PROJECT_ROOT / "Unet" / "Awesome-U-Net"
sys.path.insert(0, str(AWESOME_UNET_ROOT))
print("✅ Added UNet path:", AWESOME_UNET_ROOT)

from models.unet import UNet
from models.attunet import AttU_Net
from models.unetpp import NestedUNet

# SMP DeeplabV3+
import segmentation_models_pytorch as smp

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


# ============================================================
# Dataset
# ============================================================

class SemanticSegDataset(Dataset):
    def __init__(self, data_root, split="test", img_size=640):
        self.img_dir = Path(data_root) / "images" / split
        self.mask_dir = Path(data_root) / "masks" / split
        assert self.img_dir.exists(), f"Missing: {self.img_dir}"
        assert self.mask_dir.exists(), f"Missing: {self.mask_dir}"

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

        print(f"[{split}] samples = {len(self.samples)}  img_size = {self.img_size}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p = self.samples[idx]

        bgr = cv2.imread(str(p["img"]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        mask_img = cv2.imread(str(p["mask"]), cv2.IMREAD_GRAYSCALE)

        rgb = cv2.resize(rgb, (self.img_size, self.img_size))
        mask = cv2.resize(mask_img, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

        rgb = rgb.astype(np.float32) / 255.0
        rgb = rgb.transpose(2, 0, 1)  # (C,H,W)

        return {
            "image": torch.from_numpy(rgb),
            "mask": torch.from_numpy(mask).unsqueeze(0),  # (1,H,W)
            "name": p["name"],
        }


# ============================================================
# Metrics
# ============================================================

def compute_metrics(pred, gt):
    """
    pred, gt: numpy float or bool, shape (H,W)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    union = (pred | gt).sum()

    iou = tp / (union + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    return iou, dice, prec, rec


# ============================================================
# 构建模型
# ============================================================

def build_unet(net_key, in_channels=3, out_channels=1):
    n = net_key.lower()

    if n == "unet":
        return UNet(in_channels, out_channels)

    if n == "attenunet":
        return AttU_Net(in_channels, out_channels)

    if n in ["unet++", "unetpp", "nestedunet"]:
        return NestedUNet(num_classes=out_channels, input_channels=in_channels)

    raise ValueError(f"Unknown UNet-like net: {net_key}")


def build_deeplabv3plus(encoder_name="resnet50",
                        in_channels=3,
                        out_channels=1):
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=None,   # 由我们自己加载 ckpt 覆盖
        in_channels=in_channels,
        classes=out_channels
    )
    return model


# ============================================================
# 推理 + 保存 mask & overlay
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, mask_dir: Path, overlay_dir: Path):
    model.eval()

    n = 0
    sum_iou = sum_dice = sum_prec = sum_rec = 0.0

    for batch in tqdm(loader, desc="推理中"):
        img = batch["image"].to(device)   # (B,3,H,W)
        gt = batch["mask"].to(device)     # (B,1,H,W)
        names = batch["name"]

        out = model(img)
        # 兼容 dict 输出（torchvision 风格），SMP 是直接 tensor
        if isinstance(out, dict):
            logits = out["out"]
        else:
            logits = out  # (B,1,H,W)

        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()       # (B,1,H,W)

        B = img.shape[0]
        for i in range(B):
            pm = pred[i, 0].cpu().numpy()  # (H,W)
            gm = gt[i, 0].cpu().numpy()

            # 计算指标
            iou, dice, prec, rec = compute_metrics(pm, gm)
            sum_iou += iou
            sum_dice += dice
            sum_prec += prec
            sum_rec += rec
            n += 1

            # 保存黑白 mask
            mask_path = mask_dir / f"{names[i]}.png"
            cv2.imwrite(str(mask_path), (pm * 255).astype(np.uint8))

            # 保存 overlay
            rgb = (img[i].cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)  # (H,W,3)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay = bgr.copy()
            m_bool = pm.astype(bool)
            overlay[m_bool] = (
                overlay[m_bool] * 0.6 + np.array([255, 0, 0]) * 0.4
            )
            overlay = overlay.astype(np.uint8)
            overlay_path = overlay_dir / f"{names[i]}.png"
            cv2.imwrite(str(overlay_path), overlay)

    miou = sum_iou / max(n, 1)
    mdice = sum_dice / max(n, 1)
    mprec = sum_prec / max(n, 1)
    mrec = sum_rec / max(n, 1)

    return miou, mdice, mprec, mrec


# ============================================================
# Main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser("UNet & SMP DeepLabV3+ Inference")

    ap.add_argument("--data_root", default="/root/autodl-tmp/dataset_seg",
                    help="包含 images/ & masks/ 的根目录")
    ap.add_argument("--weights_root", default="/root/autodl-tmp/Unet/unet_weights",
                    help="各模型权重根目录: {weights_root}/{net}/best.pth")
    ap.add_argument("--net", type=str, default="unet++",
                    help="unet / attenunet / unet++ / deeplabv3plus")
    # encoder/backbone（兼容两种写法）
    ap.add_argument("--encoder", "--backbone", dest="encoder",
                    type=str, default="resnet50",
                    help="DeepLabV3+ 的 encoder 名称，如 resnet50 / resnet101 / mit_b3 等")
    ap.add_argument("--split", type=str, default="test",
                    help="使用哪个子集: train / valid / test 等")
    ap.add_argument("--img_size", type=int, default=256,
                    help="推理解码时统一 resize 的尺寸")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_root", default="/root/autodl-tmp/results",
                    help="推理结果根目录，下面会按 net 名创建子目录")
    return ap.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset + Loader
    ds = SemanticSegDataset(args.data_root, split=args.split, img_size=args.img_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # 输出目录: /out_root/{net}/mask, overlay
    net_name = args.net.lower()
    out_dir = Path(args.out_root) / net_name
    mask_dir = out_dir / "mask"
    overlay_dir = out_dir / "overlay"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    print("结果输出目录:", out_dir)

    # 构建模型
    if net_name == "deeplabv3plus":
        print("==> 使用 SMP DeepLabV3Plus, encoder =", args.encoder)
        model = build_deeplabv3plus(
            encoder_name=args.encoder,
            in_channels=3,
            out_channels=1
        )
    else:
        print("==> 使用 UNet 系列模型:", net_name)
        model = build_unet(net_name, in_channels=3, out_channels=1)

    model = model.to(device)

    # 加载权重
    ckpt_path = Path(args.weights_root) / net_name / "best.pth"
    print("加载权重:", ckpt_path)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"State_dict loaded. Missing={len(missing)}, Unexpected={len(unexpected)}")
    if len(missing):
        print("  Missing keys (前几个):", missing[:5])
    if len(unexpected):
        print("  Unexpected keys (前几个):", unexpected[:5])

    # 评估 & 保存预测
    miou, mdice, mprec, mrec = evaluate(model, loader, device, mask_dir, overlay_dir)

    print("\n======== 评估结果 ========")
    print("模型   :", net_name)
    if net_name == "deeplabv3plus":
        print("Encoder:", args.encoder)
    print("Split  :", args.split)
    print(f"mIoU   : {miou:.4f}")
    print(f"Dice   : {mdice:.4f}")
    print(f"Prec   : {mprec:.4f}")
    print(f"Rec    : {mrec:.4f}")
    print("===========================")


if __name__ == "__main__":
    args = parse_args()
    main(args)
