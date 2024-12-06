from ultralytics import YOLO
import torch
import os

# Get the absolute path to the data.yaml file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml_path = os.path.join(current_dir, 'Valorant-Object-Detection-22', 'data.yaml')

# Verify CUDA is being used
print(f"Using CUDA: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.get_device_name(0)}")

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model with our dataset
model.train(
    data=data_yaml_path,  # use absolute path to data.yaml
    epochs=100,  # increased epochs since we have GPU acceleration
    imgsz=640,  # image size
    batch=32,    # increased batch size for A100
    device=0,    # use GPU
    workers=8,   # number of worker threads
    name='valorant_model',  # save results to runs/detect/valorant_model
    patience=50,  # early stopping patience
    save=True,   # save checkpoints
    cache=True   # cache images for faster training
)

