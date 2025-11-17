import os
import time
import argparse
import random
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


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


def extract_box_from_mask(mask):
    """
    从语义 mask 中提取 xyxy 外接框 (像素坐标, 左上闭, 右下开)
    mask: (H, W) 0/1 或 0/255
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        # 没有前景，给一个极小的框，防止崩溃
        return [0, 0, 1, 1]
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


def bce_dice_loss(logits, target):
    """
    logits: (B,1,256,256)
    target: (B,1,256,256) 0/1
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


# ============================================================
# Dataset: 语义分割 + Box Prompt
# ============================================================

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


class SemanticBoxPromptDataset(Dataset):
    """
    语义 mask (前景/背景) + 从语义 mask 提取外接框作为 box prompt

    ROOT/
      images/{split}/xxx.jpg
      masks/{split}/xxx.png  # 单通道 0/255

    任务: class-agnostic 语义分割（前景 vs 背景）
    """

    def __init__(self, data_root, split, img_size=1024):
        self.img_dir = Path(data_root) / "images" / split
        self.mask_dir = Path(data_root) / "masks" / split
        assert self.img_dir.exists(), f"{self.img_dir} 不存在"
        assert self.mask_dir.exists(), f"{self.mask_dir} 不存在"

        self.img_size = img_size
        self.resize = ResizeLongestSide(self.img_size)

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

        print(f"[{split}] samples = {len(self.samples)} (语义前景掩码 + Box Prompt)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p = self.samples[idx]

        # 读图 (H,W,3) BGR -> RGB
        bgr = cv2.imread(str(p["img"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"读取图像失败: {p['img']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        # 读语义 mask
        mask_img = cv2.imread(str(p["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise RuntimeError(f"读取mask失败: {p['mask']}")
        # 若大小不一致，先 resize 到与原图相同
        if mask_img.shape[:2] != (H, W):
            mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_gt = (mask_img > 0).astype(np.uint8)  # 0/1

        # ========= 图像 resize + pad =========
        img_res = self.resize.apply_image(rgb)    # (new_h,new_w,3)
        new_h, new_w = img_res.shape[:2]
        pad_h, pad_w = self.img_size - new_h, self.img_size - new_w

        img_t = torch.as_tensor(img_res).permute(2, 0, 1).float()  # (3, new_h, new_w)
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        img_t = (img_t - mean) / std
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h))  # (3,1024,1024)

        # ========= mask -> 1024 / 256 =========
        # 先跟随图像缩放再 pad
        mask_res = cv2.resize(mask_gt, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask1024 = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        mask1024[:new_h, :new_w] = mask_res
        mask256 = cv2.resize(mask1024, (256, 256), interpolation=cv2.INTER_NEAREST)
        # 转成 0/1
        mask1024 = (mask1024 > 0).astype(np.uint8)
        mask256 = (mask256 > 0).astype(np.uint8)

        # ========= 生成 BOX prompt（语义前景的外接框）=========
        # 注意: 这里是在原图分辨率上提 box，再映射到 resize 后的坐标系
        xyxy = extract_box_from_mask(mask_gt)  # [x1,y1,x2,y2] in 原图
        box_arr = np.array([xyxy], dtype=np.float32)

        box_resized = self.resize.apply_boxes(box_arr, (H, W))[0]
        box_resized[[0, 2]] = np.clip(box_resized[[0, 2]], 0, new_w - 1)
        box_resized[[1, 3]] = np.clip(box_resized[[1, 3]], 0, new_h - 1)

        return {
            "image": img_t,                                # (3,1024,1024)
            "box_prompt": torch.from_numpy(box_resized),   # (4,)
            "gt256": torch.from_numpy(mask256).float(),    # (256,256)
            "gt1024": mask1024,                            # numpy (1024,1024)
            "name": p["name"],
        }


def collate_fn(batch):
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    boxes = torch.stack([b["box_prompt"] for b in batch], dim=0)  # (B,4)
    gt256 = torch.stack([b["gt256"] for b in batch], dim=0).unsqueeze(1)
    gt1024 = [b["gt1024"] for b in batch]
    names = [b["name"] for b in batch]

    return {
        "image": imgs,
        "box_prompt": boxes,
        "gt256": gt256,
        "gt1024": gt1024,
        "name": names,
    }


# ============================================================
# Evaluate
# ============================================================

@torch.no_grad()
def evaluate(sam, loader, device, amp=True, max_batches=50):
    """
    在验证集上计算 mIoU (基于 1024x1024 的语义 mask)
    """
    sam.eval()
    image_pe = sam.prompt_encoder.get_dense_pe().to(device)

    tot_iou, cnt = 0.0, 0

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break

        img = batch["image"].to(device)            # (B,3,1024,1024)
        boxes = batch["box_prompt"].to(device)     # (B,4)
        gt_list = batch["gt1024"]                 # list of (1024,1024) numpy

        B = img.shape[0]

        for i in range(B):
            with torch.cuda.amp.autocast(enabled=amp):
                emb = sam.image_encoder(img[i:i+1])   # (1,C,H',W')
                sp, dn = sam.prompt_encoder(
                    points=None,
                    boxes=boxes[i:i+1].unsqueeze(1),  # (1,1,4)
                    masks=None
                )
                logits, _ = sam.mask_decoder(
                    image_embeddings=emb,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sp,
                    dense_prompt_embeddings=dn,
                    multimask_output=False
                )

            prob = torch.sigmoid(
                F.interpolate(logits, size=(1024, 1024),
                              mode="bilinear", align_corners=False)
            ).squeeze().cpu().numpy()  # (H,W)

            pm = (prob > 0.5).astype(np.uint8)
            gm = gt_list[i].astype(np.uint8)

            tot_iou += compute_iou(pm, gm)
            cnt += 1

    sam.train()
    return tot_iou / max(1, cnt)


# ============================================================
# Train
# ============================================================

def train_decoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    set_seed(args.seed)
    ensure_dir(Path(args.save_dir))

    # ===== Dataset =====
    train_ds = SemanticBoxPromptDataset(args.data_root, "train", img_size=args.img_size)
    val_ds = SemanticBoxPromptDataset(args.data_root, "valid", img_size=args.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # ===== Load SAM =====
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_ckpt).to(device)

    # 冻结 image_encoder & prompt_encoder，只训练 mask_decoder（轻量）
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False
    sam.image_encoder.eval()
    sam.prompt_encoder.eval()
    sam.mask_decoder.train()

    opt = torch.optim.AdamW(
        sam.mask_decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    image_pe = sam.prompt_encoder.get_dense_pe().to(device)

    best_miou = 0.0

    # ============================================================
    #            Train Loop
    # ============================================================
    for epoch in range(args.epochs):
        total_loss = 0.0
        t0 = time.time()
        sam.mask_decoder.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            img = batch["image"].to(device)                # (B,3,1024,1024)
            boxes = batch["box_prompt"].to(device)         # (B,4)
            gt = batch["gt256"].to(device)                 # (B,1,256,256)
            B = img.shape[0]

            with torch.no_grad():
                emb = sam.image_encoder(img)               # (B,C,H',W')

            logits_list = []

            for i in range(B):
                with torch.no_grad():
                    sp, dn = sam.prompt_encoder(
                        points=None,
                        boxes=boxes[i:i+1].unsqueeze(1),   # (1,1,4)
                        masks=None
                    )

                with torch.cuda.amp.autocast(enabled=args.amp):
                    logit_i, _ = sam.mask_decoder(
                        image_embeddings=emb[i:i+1],
                        image_pe=image_pe,
                        sparse_prompt_embeddings=sp,
                        dense_prompt_embeddings=dn,
                        multimask_output=False
                    )
                logits_list.append(logit_i)

            logits = torch.cat(logits_list, dim=0)         # (B,1,256,256)

            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = bce_dice_loss(logits, gt)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = total_loss / len(train_loader)

        # ====== Validation ======
        miou = evaluate(
            sam,
            val_loader,
            device,
            amp=args.amp,
            max_batches=args.val_max_batches
        )

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss={train_loss:.4f} | mIoU={miou:.4f} | "
              f"Time={time.time()-t0:.1f}s")

        # 保存 checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "mask_decoder": sam.mask_decoder.state_dict(),
            "optimizer": opt.state_dict(),
            "miou": miou,
        }
        torch.save(ckpt, str(Path(args.save_dir) / f"decoder_epoch{epoch+1:03d}.pt"))

        if miou > best_miou:
            best_miou = miou
            ckpt["best_miou"] = best_miou
            torch.save(ckpt, str(Path(args.save_dir) / "decoder_best.pt"))
            print(f"🎉 New Best mIoU = {best_miou:.4f}")

    print("训练完成！模型保存在:", args.save_dir)


# ============================================================
# Main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser("SAM Fine-tuning for Semantic Segmentation (Foreground/Background)")

    ap.add_argument("--data_root", default="/root/autodl-tmp/dataset_seg")
    ap.add_argument("--sam_ckpt", default="/root/autodl-tmp/segment-anything/sam_vit_b_01ec64.pth")
    ap.add_argument("--model_type", default="vit_b")
    ap.add_argument("--save_dir", default="/root/autodl-tmp/runs/sam_semantic_box")

    ap.add_argument("--img_size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--val_max_batches", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("使用语义分割 GT Mask + Box Prompt 微调 SAM Mask Decoder")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 60)
    train_decoder(args)
