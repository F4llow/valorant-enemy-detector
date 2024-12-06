from flask import Flask, request, render_template, send_file
import os
from predict import predict_image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Ensure model exists
def ensure_model():
    model_dir = 'model/weights'
    model_path = os.path.join(model_dir, 'best.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please ensure the model file exists.")
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
            
            # Move the result to results folder with unique name
            result_filename = f'result_{filename}'
            result_path = os.path.join(RESULTS_FOLDER, result_filename)
            os.replace('predicted_image.jpg', result_path)
            
            # Return the template with the result image and stats
            return render_template('result.html', 
                                result_image=f'results/{result_filename}',
                                original_image=f'uploads/{filename}',
                                stats=stats)
    
    return render_template('upload.html')

if __name__ == '__main__':
    # Ensure model is downloaded
    ensure_model()
    # Run the app
    app.run(debug=True, port=5000)
