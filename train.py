from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=4,
    workers=8,
    lr0=0.005,
    patience=30,
    augment=True,
    device='cpu',
    project='runs',
    name='train_v2',
)