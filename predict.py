from ultralytics import YOLO
import cv2
import time
from datetime import datetime
import os

def predict_image(model_path, image_path):
    stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'detections': [],
        'timing': {},
        'image_info': {},
        'model_info': {}
    }
    
    start_time = time.time()
    model = YOLO(model_path)
    stats['timing']['model_load'] = f"{(time.time() - start_time):.2f}s"
    stats['model_info']['model_path'] = model_path
    
    start_time = time.time()
    img = cv2.imread(image_path)
    original_shape = img.shape
    stats['image_info']['original_size'] = f"{original_shape[1]}x{original_shape[0]}"
    stats['image_info']['path'] = image_path
    stats['timing']['image_load'] = f"{(time.time() - start_time):.2f}s"
    
    start_time = time.time()
    results = model.predict(image_path, conf=0.25, save=False)
    inference_time = time.time() - start_time
    stats['timing']['inference'] = f"{inference_time:.2f}s"
    
    start_time = time.time()
    for r in results:
        stats['image_info']['model_input_size'] = f"{r.orig_shape[1]}x{r.orig_shape[0]}"
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            stats['detections'].append({
                'class': class_name,
                'confidence': f"{conf:.2%}",
                'position': f"({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})"
            })
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            label = f'{class_name} {conf:.2f}'
            cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    stats['timing']['drawing'] = f"{(time.time() - start_time):.2f}s"
    
    start_time = time.time()
    output_path = 'predicted_image.jpg'
    cv2.imwrite(output_path, img)
    stats['timing']['saving'] = f"{(time.time() - start_time):.2f}s"
    
    stats['summary'] = {
        'total_detections': len(stats['detections']),
        'fps': f"{1/inference_time:.1f}"
    }
    
    return stats
