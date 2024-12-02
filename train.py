from ultralytics import YOLO
import torch

def main():
    # Print GPU availability info
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.get_device_name()}")

    # Initialize the model
    model = YOLO('yolov8n.pt')  # load a pretrained model

    # Train the model
    results = model.train(
        data='C:/Users/NCallabresi/Documents/ValorantAgentsDataset2/v8/Valorant-Object-Detection-22/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

if __name__ == '__main__':
    main()
