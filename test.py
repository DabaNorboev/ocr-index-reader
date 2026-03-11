from ultralytics import YOLO

model = YOLO(r'runs/detect/runs/train_v2/weights/best.pt')

metrics = model.val(
    data='dataset.yaml',
    split='test',
    conf=0.5,
)

print("\n=== РЕЗУЛЬТАТЫ НА TEST ===")
print(f"mAP50:      {metrics.box.map50:.3f}")
print(f"mAP50-95:   {metrics.box.map:.3f}")
print(f"Precision:  {metrics.box.mp:.3f}")
print(f"Recall:     {metrics.box.mr:.3f}")