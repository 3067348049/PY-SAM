
import os
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


# ============================================================
# Utils
# ============================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def extract_box_from_mask(mask):
    """
    从语义 mask 中提取 xyxy 外接框
    mask: (H,W) 0/1 或 0/255
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 1, 1]
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


def compute_metrics(pred, gt):
    """
    pred, gt: (H,W) 0/1
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    tp = inter
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()

    return dict(
        iou=inter / (union + 1e-6),
        dice=(2 * inter) / (pred.sum() + gt.sum() + 1e-6),
        precision=tp / (tp + fp + 1e-6),
        recall=tp / (tp + fn + 1e-6)
    )


# ============================================================
# Dataset (test split)
# ============================================================

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


class SemanticTestDataset(Dataset):
    """
    目录结构：
    ROOT/
      images/test/xxx.jpg|png
      masks/test/xxx.png

    推理时：
      - 读 test 图像 & 语义 mask
      - 从语义 mask 提取外接框作为 box prompt
      - 图像和 mask 统一到 1024（或 args.img_size）
    """

    def __init__(self, data_root, img_size=1024):
        self.img_dir = Path(data_root) / "images" / "test"
        self.mask_dir = Path(data_root) / "masks" / "test"
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

        print(f"[test] samples = {len(self.samples)} (语义前景掩码 + Box Prompt)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p = self.samples[idx]

        # 读图 BGR->RGB
        bgr = cv2.imread(str(p["img"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"读取图像失败: {p['img']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        # 读 mask
        mask_img = cv2.imread(str(p["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise RuntimeError(f"读取mask失败: {p['mask']}")
        if mask_img.shape[:2] != (H, W):
            mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_gt = (mask_img > 0).astype(np.uint8)  # 0/1

        # ========= 图像 resize + pad =========
        img_res = self.resize.apply_image(rgb)  # (new_h,new_w,3)
        new_h, new_w = img_res.shape[:2]
        pad_h, pad_w = self.img_size - new_h, self.img_size - new_w

        img_t = torch.as_tensor(img_res).permute(2, 0, 1).float()
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        img_t = (img_t - mean) / std
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h))  # (3,1024,1024)

        # ========= mask -> 1024 =========
        mask_res = cv2.resize(mask_gt, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask1024 = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        mask1024[:new_h, :new_w] = mask_res
        mask1024 = (mask1024 > 0).astype(np.uint8)

        # ========= box prompt (从语义 mask 提外接框) =========
        xyxy = extract_box_from_mask(mask_gt)  # 原图坐标
        box_arr = np.array([xyxy], dtype=np.float32)
        box_resized = self.resize.apply_boxes(box_arr, (H, W))[0]
        box_resized[[0, 2]] = np.clip(box_resized[[0, 2]], 0, new_w - 1)
        box_resized[[1, 3]] = np.clip(box_resized[[1, 3]], 0, new_h - 1)

        return {
            "image": img_t,
            "box_prompt": torch.from_numpy(box_resized),
            "gt1024": mask1024,
            "rgb": rgb,
            "name": p["name"],
            "original_size": (H, W),
            "resized_size": (new_h, new_w),
        }


def collate_fn(batch):
    # 我们这里就是 batch_size=1，用 list 存一些变长元素比较省事
    return {
        "image": torch.stack([b["image"] for b in batch]),                # (B,3,1024,1024)
        "box_prompt": torch.stack([b["box_prompt"] for b in batch]),      # (B,4)
        "gt1024": [b["gt1024"] for b in batch],
        "rgb": [b["rgb"] for b in batch],
        "name": [b["name"] for b in batch],
        "original_size": [b["original_size"] for b in batch],
        "resized_size": [b["resized_size"] for b in batch],
    }


# ============================================================
# Inference（只保存 mask + overlay）
# ============================================================

@torch.no_grad()
def inference_with_box(sam,
                       loader,
                       device,
                       mask_dir: Path,
                       overlay_dir: Path,
                       color_rgb=(255, 0, 0),
                       model_name="SAM"):
    """
    - sam: SAM 模型（原始或微调）
    - mask_dir: 保存二值 mask 的目录
    - overlay_dir: 保存 overlay 的目录
    - color_rgb: 覆盖颜色（RGB）
    """
    sam.eval()
    ensure_dir(mask_dir)
    ensure_dir(overlay_dir)

    image_pe = sam.prompt_encoder.get_dense_pe().to(device)
    all_metrics = []

    for batch in tqdm(loader, desc=f"Inference [{model_name}]"):
        img = batch["image"].to(device)                    # (1,3,1024,1024)
        box = batch["box_prompt"].to(device).unsqueeze(1)  # (1,1,4)

        gt = batch["gt1024"][0]                            # (1024,1024)
        name = batch["name"][0]
        rgb = batch["rgb"][0]
        H, W = batch["original_size"][0]
        new_h, new_w = batch["resized_size"][0]

        # ===== SAM 前向 =====
        emb = sam.image_encoder(img)
        sp, dn = sam.prompt_encoder(points=None, boxes=box, masks=None)
        logits, _ = sam.mask_decoder(
            image_embeddings=emb,
            image_pe=image_pe,
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=dn,
            multimask_output=False
        )

        prob = torch.sigmoid(
            F.interpolate(
                logits,
                (gt.shape[0], gt.shape[1]),  # 1024x1024
                mode="bilinear",
                align_corners=False
            )
        ).squeeze().cpu().numpy()  # (1024,1024)

        pred_1024 = (prob > 0.5).astype(np.uint8)

        # 裁掉 pad，再 resize 回原图大小
        pred_crop = pred_1024[:new_h, :new_w]
        pred_orig = cv2.resize(pred_crop, (W, H), interpolation=cv2.INTER_NEAREST)
        gt_orig = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        gt_orig = (gt_orig > 0).astype(np.uint8)

        # ===== 计算指标 =====
        metrics = compute_metrics(pred_orig, gt_orig)
        all_metrics.append(metrics)

        # ===== 保存黑白 mask =====
        cv2.imwrite(str(mask_dir / f"{name}.png"), pred_orig * 255)

        # ===== 保存 overlay（在 RGB 空间画，再转 BGR 存盘） =====
        overlay = rgb.copy()
        m_bool = pred_orig.astype(bool)
        overlay[m_bool] = (
            0.6 * overlay[m_bool] + 0.4 * np.array(color_rgb)
        ).astype(np.uint8)

        cv2.imwrite(
            str(overlay_dir / f"{name}.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

    # ===== 统计平均指标（终端打印即可，不保存图） =====
    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    print(f"\n=== {model_name} 平均指标 ===")
    for k, v in avg.items():
        print(f"  {k:9s}: {v:.4f}")
    print("==========================\n")

    return avg


# ============================================================
# Main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser("Compare Original SAM vs Semantic Fine-tuned SAM (Box Prompt, Simple Output)")

    ap.add_argument("--data_root", default="/root/autodl-tmp/dataset_seg")
    ap.add_argument("--sam_ckpt", default="/root/autodl-tmp/segment-anything/sam_vit_b_01ec64.pth")
    ap.add_argument("--finetuned_ckpt", default="/root/autodl-tmp/runs/sam_semantic_box/decoder_best.pt")
    ap.add_argument("--model_type", default="vit_b")
    ap.add_argument("--img_size", type=int, default=1024)
    ap.add_argument("--results_root", default="/root/autodl-tmp/results",
                    help="根目录，下面会创建 sam/ 和 ft_sam/")

    return ap.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset + Loader
    test_ds = SemanticTestDataset(args.data_root, img_size=args.img_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    results_root = Path(args.results_root)
    sam_root = results_root / "sam"
    ft_sam_root = results_root / "ft_sam"

    print("\n加载模型...")
    sam_orig = sam_model_registry[args.model_type](
        checkpoint=args.sam_ckpt
    ).to(device)

    sam_ft = sam_model_registry[args.model_type](
        checkpoint=args.sam_ckpt
    ).to(device)
    ckpt = torch.load(args.finetuned_ckpt, map_location=device)
    sam_ft.mask_decoder.load_state_dict(ckpt["mask_decoder"])
    print("已加载微调后的 mask_decoder 权重")

    # ========= 原始 SAM =========
    print("\n推理原始 SAM...")
    avg_orig = inference_with_box(
        sam_orig,
        test_loader,
        device,
        mask_dir=sam_root / "mask",
        overlay_dir=sam_root / "overlay",
        color_rgb=(0, 0, 255),   # 蓝色
        model_name="Original SAM"
    )

    # ========= 微调 SAM =========
    print("推理微调 SAM...")
    avg_ft = inference_with_box(
        sam_ft,
        test_loader,
        device,
        mask_dir=ft_sam_root / "mask",
        overlay_dir=ft_sam_root / "overlay",
        color_rgb=(0, 0, 255),
        model_name="Fine-tuned SAM"
    )

    print("对比完成！结果已保存到：")
    print("  原始 SAM :", sam_root)
    print("  微调 SAM :", ft_sam_root)


if __name__ == "__main__":
    main(parse_args())
