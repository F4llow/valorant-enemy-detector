from flask import Flask, request, render_template, send_file
import os
from predict import predict_image
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Download the model if it doesn't exist
def ensure_model():
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'best.pt')
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # Start with base model
        model.load('runs/detect/train2/weights/best.pt')  # Load our trained weights
        model.save(model_path)  # Save for future use
    return model_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)
            
            # Process image
            model_path = ensure_model()
            stats = predict_image(model_path, input_path)
            
            # Get the processed image path
            output_path = 'predicted_image.jpg'
            
            # Move the result to results folder with unique name
            result_filename = f'result_{filename}'
            result_path = os.path.join(RESULTS_FOLDER, result_filename)
            os.replace(output_path, result_path)
            
            # Return the template with the result image and stats
            return render_template('result.html', 
                                result_image=f'results/{result_filename}',
                                original_image=f'uploads/{filename}',
                                stats=stats)
    
    return render_template('upload.html')

# Serve static files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/results/<filename>')
def result_file(filename):
    return send_file(os.path.join(RESULTS_FOLDER, filename))

if __name__ == '__main__':
    # Ensure model is downloaded
    ensure_model()
    # Run the app
    app.run(debug=True, port=5000)
