from ultralytics import YOLO

DATA = "/root/autodl-tmp/dataset_yolo/data.yaml"
RUNS = "/root/autodl-tmp/runs"
TEST_IMAGES = "/root/autodl-tmp/dataset_yolo/test/images"

model = YOLO("yolo11n.pt")

# 2) 训练
model.train(
    data=DATA,
    epochs=100,
    imgsz=640,
    batch=-1,
    device=0,
    workers=8,
    project=RUNS,
    name="flowers_yolo11n",
)


best = f"{RUNS}/flowers_yolo11n/weights/best.pt"
model = YOLO(best)
metrics = model.val(data=DATA, device=0)

model.predict(
    source=TEST_IMAGES,
    save=True, save_txt=True, save_conf=True,
    project=RUNS, name="flowers_pred",
    device=0
)

print("Done. YOLO results at:", RUNS)
