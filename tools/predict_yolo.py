import os, json, cv2
from ultralytics import YOLO

# ===================== 配置 =====================
ROOT = "/root/autodl-tmp"
DATA_DIR = f"{ROOT}/dataset_yolo"
RUNS = f"{ROOT}/runs"
MODEL_PATH = f"{RUNS}/flowers_yolo11n/weights/best.pt"
OUT_BASE = f"{RUNS}/flowers_pred_all"      # YOLO预测结果输出目录
PROMPT_DIR = f"{RUNS}/sam_prompts"         # SAM可用的prompt输出目录
os.makedirs(PROMPT_DIR, exist_ok=True)

# ===============================================
# Step 1: 加载 YOLO 模型
# ===============================================
print("🚀 Loading YOLO model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

splits = ["train", "valid", "test"]
prompt_summary = {}

# ===============================================
# Step 2: 逐个子集进行推理
# ===============================================
for split in splits:
    img_dir = os.path.join(DATA_DIR, split, "images")
    out_dir = os.path.join(OUT_BASE, split)
    print(f"\n=== 🔍 Predicting {split} images ===")
    results = model.predict(
        source=img_dir,
        save=True,
        save_txt=True,
        save_conf=True,
        project=OUT_BASE,
        name=split,
        device=0
    )

    label_dir = os.path.join(out_dir, "labels")
    prompt_json = os.path.join(PROMPT_DIR, f"{split}.json")
    data = []

    # ===============================================
    # Step 3: YOLO TXT → SAM JSON Prompt
    # ===============================================
    for file in sorted(os.listdir(label_dir)):
        if not file.endswith(".txt"):
            continue
        label_path = os.path.join(label_dir, file)
        img_name = file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            # 尝试png扩展
            img_path = img_path.replace(".jpg", ".png")
            if not os.path.exists(img_path):
                print(f"⚠️ Missing image for {file}")
                continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            lines = f.read().strip().splitlines()
        boxes = []
        for line in lines:
            cls, cx, cy, bw, bh, conf = map(float, line.split())
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append({
                "class_id": int(cls),
                "conf": round(conf, 4),
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })

        data.append({
            "image": img_path,
            "height": h,
            "width": w,
            "objects": boxes
        })

    # 保存成 JSON 文件
    with open(prompt_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ {split} prompts saved -> {prompt_json}")
    prompt_summary[split] = len(data)

print("\n🎉 All splits processed.")
print("📂 YOLO results:", OUT_BASE)
print("📂 SAM prompts:", PROMPT_DIR)
print(json.dumps(prompt_summary, indent=2))
