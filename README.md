# Valorant Enemy Detection Model

This is a machine learning model that detects enemy agents in Valorant gameplay screenshots. It uses YOLOv8 for object detection and provides a simple web interface for testing.

## Features

- Upload any Valorant screenshot
- Get instant enemy agent detections
- View detailed detection statistics
- Side-by-side comparison of original and detected images

## How to Use

1. Click the "Choose File" button to select a Valorant screenshot
2. Click "Upload" to process the image
3. View the results with bounding boxes and detection statistics

## Technical Details

- Model: YOLOv8 Nano
- Training Dataset: Roboflow Valorant Object Detection dataset (v22)
- Web Framework: Flask

## License

Dataset: CC BY 4.0 License
