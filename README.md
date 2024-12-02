# Valorant Enemy Detection Model

A machine learning-powered web application that detects enemy agents in Valorant gameplay screenshots. Built using YOLOv8 for object detection and Flask for the web interface.

## Features

- Upload any Valorant screenshot
- Get instant enemy agent detections
- View detailed detection statistics
- Side-by-side comparison of original and detected images

## Project Structure

### Core Files
- `app.py` - Main Flask application file that handles web routes and image processing
- `predict.py` - Contains the prediction logic using YOLOv8 model
- `train.py` - Script for training the YOLOv8 model on custom Valorant dataset
- `evaluate.py` - Evaluation script to test model performance
- `requirements.txt` - List of Python dependencies

### Directories
- `templates/` - HTML templates for the web interface
  - `upload.html` - Main page with image upload form
  - `result.html` - Results page showing detections
- `models/` - Directory for storing trained models (not in repository)
  - Place your `best.pt` model file here
- `uploads/` - Temporary storage for uploaded images
- `results/` - Stores detection results and visualizations
- `runs/` - Training logs and model checkpoints
- `evaluation_results/` - Model evaluation metrics and reports

## Installation

1. Clone the repository:
```bash
git clone https://github.com/F4llow/valorant-enemy-detector.git
cd valorant-enemy-detector
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your trained model file (`best.pt`) in the `models` directory

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and go to `http://127.0.0.1:5000`

3. Upload a Valorant screenshot and view the detections

## Model Details

- Base Model: YOLOv8 Nano
- Training Dataset: Roboflow Valorant Object Detection dataset (v22)
- Input Size: 640x640 pixels
- Classes: Enemy agents

## File Storage

- Uploaded images are temporarily stored in `uploads/` directory
- Detection results are saved in `results/` directory
- Model checkpoints during training are saved in `runs/detect/`
- Evaluation metrics and reports are stored in `evaluation_results/`

## Development

- The application runs in debug mode by default
- Model predictions are processed in real-time
- Temporary files are automatically cleaned up
- Web interface is mobile-responsive

## Technical Requirements

- Python 3.11+
- CUDA-compatible GPU recommended for faster inference
- Minimum 4GB RAM
- Disk space: ~1GB (including model)

## License

Dataset: CC BY 4.0 License
