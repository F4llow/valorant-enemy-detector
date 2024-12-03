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
        epochs=9,  # Changed to 9 epochs
        imgsz=640,
        batch=16,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        project='runs/detect',  # project folder
        name='train'  # experiment name
    )

    # Save the model
    model.save('best.pt')  # Save with a specific name

if __name__ == '__main__':
    main()
