import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def collect_test_images(images_test_dir: Path):
    """收集 images/test 下所有原图，返回 {stem: Path}"""
    mapping = {}
    for p in images_test_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        mapping[p.stem] = p
    return mapping


def main(args):
    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    split = args.split

    images_test_dir = data_root / "images" / split
    if not images_test_dir.exists():
        raise FileNotFoundError(f"找不到 images/{split} 目录: {images_test_dir}")

    # 1. 收集原图
    stem2img = collect_test_images(images_test_dir)
    if not stem2img:
        raise RuntimeError(f"{images_test_dir} 下没有测试图片")

    # 2. 找出有哪些模型（包含 overlay 子目录的子文件夹）
    # 自动检测出 results/ 下所有包含 overlay/ 的模型
    detected = [p.name for p in results_root.iterdir()
                if p.is_dir() and (p / "overlay").exists()]

    # 你想要的顺序（可自由改）
    ordered_models = [
        "attenunet",
        "deeplabv3plus",
        "sam",
        "unet",
        "unet++",
        "ft_sam"
    ]

    # 按指定顺序过滤（不存在则自动跳过）
    model_names = [m for m in ordered_models if m in detected]

    print("拼接顺序：", model_names)

    if not model_names:
        raise RuntimeError(f"{results_root} 下没有包含 overlay/ 的模型目录")

    print("将按以下顺序拼接：")
    print("  原图  +  " + "  +  ".join(model_names))

    # 3. 输出目录
    out_dir = results_root / "compare_all"
    ensure_dir(out_dir)
    print("输出目录:", out_dir)

    # 4. 对每个样本做拼接
    for stem, orig_path in sorted(stem2img.items()):
        # 4.1 读原图
        img_orig = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
        if img_orig is None:
            print(f"⚠ 读取原图失败: {orig_path}，跳过")
            continue
        h0, w0 = img_orig.shape[:2]

        imgs = [img_orig]

        # 4.2 依次读取每个模型的 overlay
        for net in model_names:
            ov_path = results_root / net / "overlay" / f"{stem}.png"
            if not ov_path.exists():
                print(f"  [{net}] 找不到 overlay: {ov_path}")
                continue

            img_ov = cv2.imread(str(ov_path), cv2.IMREAD_COLOR)
            if img_ov is None:
                print(f"  [{net}] 读取 overlay 失败: {ov_path}")
                continue

            # 尺寸对齐到原图
            h1, w1 = img_ov.shape[:2]
            if (h1, w1) != (h0, w0):
                img_ov = cv2.resize(img_ov, (w0, h0), interpolation=cv2.INTER_LINEAR)

            imgs.append(img_ov)

        if len(imgs) <= 1:
            # 只有原图，没有任何模型 overlay
            print(f"⚠ {stem} 没有任何模型的 overlay，跳过")
            continue

        # 4.3 横向拼接：左原图，右各个模型 overlay
        concat = np.concatenate(imgs, axis=1)

        out_path = out_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), concat)

    print("全部完成！")


def parse_args():
    ap = argparse.ArgumentParser("拼接原图与各模型 overlay 进行对比")
    ap.add_argument("--data_root", default="/root/autodl-tmp/dataset_seg",
                    help="包含 images/<split> 的数据根目录")
    ap.add_argument("--results_root", default="/root/autodl-tmp/results",
                    help="包含各模型结果子目录的根目录")
    ap.add_argument("--split", default="test",
                    help="使用哪个子集（一般就是 test）")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
