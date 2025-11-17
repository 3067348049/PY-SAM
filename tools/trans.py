import os
import json
from pathlib import Path

import numpy as np
import cv2
from collections import defaultdict


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def polys_to_mask(segmentation, h, w):
    """
    COCO polygon segmentation -> (h, w) uint8 mask
    segmentation: list of [x1,y1,x2,y2,...] 或 list of 多个 polygon
    返回: 0/255 的 uint8 mask
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return mask

    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        # 防止越界
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)

    return mask


def process_split(root: Path, split: str):
    img_dir = root / "images" / split
    anno_path = img_dir / "_annotations.coco.json"
    out_dir = root / "masks" / split
    ensure_dir(out_dir)

    if not anno_path.exists():
        print(f"[WARN] {anno_path} 不存在，跳过 {split}")
        return

    print(f"\n=== 处理 split = {split} ===")
    print(f"读取标注: {anno_path}")

    with open(anno_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    # 建立 image_id -> image_info 映射
    img_dict = {im["id"]: im for im in images}

    # image_id -> 该图下的所有 annotation
    ann_by_img = defaultdict(list)
    for ann in anns:
        img_id = ann["image_id"]
        ann_by_img[img_id].append(ann)

    print(f"共有 images: {len(images)}, annotations: {len(anns)}")

    num_masks = 0

    for img in images:
        img_id = img["id"]
        H = img["height"]
        W = img["width"]
        file_name = img["file_name"]
        stem = Path(file_name).stem

        this_anns = ann_by_img.get(img_id, [])
        if len(this_anns) == 0:
            # 没有标注也生成一张全 0 的 mask，防止后续代码崩
            mask = np.zeros((H, W), dtype=np.uint8)
        else:
            # 合并该图所有实例的 polygon 到同一张 mask
            mask = np.zeros((H, W), dtype=np.uint8)
            for ann in this_anns:
                seg = ann.get("segmentation", [])
                inst_mask = polys_to_mask(seg, H, W)
                # 合并前景
                mask[inst_mask > 0] = 255

        out_path = out_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), mask)
        num_masks += 1

        if num_masks <= 3:
            print(f"  示例保存: {out_path}")

    print(f"[{split}] 完成，共生成 {num_masks} 张 mask，保存在: {out_dir}")


def main():
    ROOT = Path("/root/autodl-tmp/dataset_seg")

    splits = ["train", "valid", "test"]

    for sp in splits:
        process_split(ROOT, sp)


if __name__ == "__main__":
    main()
