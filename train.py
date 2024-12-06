from ultralytics import YOLO
import torch
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml_path = os.path.join(current_dir, 'Valorant-Object-Detection-22', 'data.yaml')

print(f"Using CUDA: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.get_device_name(0)}")

model = YOLO('yolov8n.pt')

model.train(
    data=data_yaml_path,
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    workers=8,
    name='valorant_model',
    patience=50,
    save=True,
    cache=True
)

