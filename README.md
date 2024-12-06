# Valorant Enemy Detector 🎮

A Flask-based web application that uses YOLOv8 to detect enemies in Valorant gameplay screenshots. The model has been trained on a custom Valorant dataset from Roboflow.

## 🚀 Features

- Real-time enemy detection in Valorant screenshots
- Detailed detection statistics and timing breakdown
- Modern, responsive web interface with Valorant-themed styling
- Comprehensive detection information including:
  - Confidence scores
  - Bounding box positions
  - Processing speed (FPS)
  - Image size information

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)
- Required Python packages listed in `requirements.txt`

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/F4llow/valorant-enemy-detector.git
cd valorant-enemy-detector
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Ensure the model file exists at `model/weights/best.pt`

## 🎮 Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload a Valorant screenshot and click "Detect Enemies"

## 📁 Project Structure

```
valorant-enemy-detector/
├── app.py              # Main Flask application
├── predict.py          # YOLOv8 prediction logic
├── train.py           # Training script for YOLOv8 model
├── download.py        # Dataset download script
├── requirements.txt    # Python dependencies
├── model/
│   └── weights/
│       └── best.pt     # Trained YOLOv8 model
├── static/
│   ├── uploads/        # Uploaded images (gitignored)
│   └── results/        # Detection results (gitignored)
└── templates/
    ├── upload.html     # Upload page template
    └── result.html     # Results page template
```

## 🔍 Detection Details

The application provides comprehensive information about each detection:

- **Detection Statistics**
  - Total number of detections
  - Processing speed in FPS
  - Timestamp of detection

- **Image Information**
  - Original image dimensions
  - Model input dimensions
  - File path information

- **Timing Breakdown**
  - Model loading time
  - Image preprocessing time
  - Inference time
  - Drawing and saving time

- **Detection Results**
  - Class name
  - Confidence score
  - Bounding box coordinates

## 🤖 Training the Model

The model was trained using YOLOv8 on a custom Valorant dataset. The training process is handled by `train.py` and was performed on William & Mary's high-performance computing cluster (bg13.cs.wm.edu) using an NVIDIA A100 GPU for optimal performance.

```python
# Key training parameters
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
```

Training infrastructure:
- Hardware: NVIDIA A100 GPU (40GB VRAM)
- Environment: William & Mary's bg13.cs.wm.edu computing cluster
- Framework: PyTorch with CUDA acceleration

The training script includes:
- Automatic CUDA detection and utilization
- Early stopping for optimal model selection
- Image caching for faster training
- Checkpoint saving

## 📥 Dataset

The dataset was managed using Roboflow and can be downloaded using the provided `download.py` script. The dataset includes:
- Valorant gameplay screenshots
- Annotated enemy positions
- Various game scenarios and maps

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Flask web framework
- Roboflow for dataset management
- CUDA for GPU acceleration
