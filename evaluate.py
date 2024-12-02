from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm
import glob

def evaluate_model(model_path, test_dir):
    # Load the trained model
    model = YOLO(model_path)
    
    # Get all images in test directory
    test_images = glob.glob(os.path.join(test_dir, '*.jpg'))
    total_images = len(test_images)
    print(f"\nFound {total_images} test images")
    
    # Create output directory for annotated images
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Track detection statistics
    total_detections = 0
    detection_counts = {}  # To track counts per class
    
    # Process images in batches of 32
    batch_size = 32
    
    # Create progress bar
    pbar = tqdm(total=total_images, desc="Processing images")
    
    for i in range(0, total_images, batch_size):
        batch_images = test_images[i:i + batch_size]
        
        # Run inference on batch
        results = model.predict(batch_images, conf=0.25, save=False, verbose=False)
        
        # Process each image in the batch
        for img_path, result in zip(batch_images, results):
            boxes = result.boxes
            total_detections += len(boxes)
            
            if len(boxes) > 0:  # Only process and save images with detections
                # Load and annotate image
                img = cv2.imread(img_path)
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = box.cls[0]
                    class_name = model.names[int(cls)]
                    
                    # Update class counts
                    detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
                    
                    # Draw box
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Put class name and confidence
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(img, label, (int(x1), int(y1)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save annotated image
                output_path = os.path.join(output_dir, os.path.basename(img_path))
                cv2.imwrite(output_path, img)
            
            # Update progress bar
            pbar.update(1)
            
            # Print interim statistics every 100 images
            if pbar.n % 100 == 0:
                print(f"\nInterim Statistics at {pbar.n}/{total_images} images:")
                print(f"Total detections so far: {total_detections}")
                print("Top 5 detected classes:")
                top_classes = sorted(detection_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for class_name, count in top_classes:
                    print(f"{class_name}: {count}")
                print()
    
    pbar.close()
    
    # Print final summary statistics
    print("\nFinal Evaluation Summary:")
    print(f"Total images processed: {total_images}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/total_images:.2f}")
    print("\nAll detections per class:")
    for class_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{class_name}: {count}")
    
    # Save only images with detections
    print(f"\nAnnotated images with detections have been saved to: {output_dir}")
    print(f"Number of images with detections: {len(os.listdir(output_dir))}")

if __name__ == '__main__':
    # Paths
    model_path = 'runs/detect/train2/weights/best.pt'  # using the weights from our 8-epoch training
    test_dir = 'C:/Users/NCallabresi/Documents/ValorantAgentsDataset2/v8/Valorant-Object-Detection-22/test/images'
    
    # Run evaluation
    evaluate_model(model_path, test_dir)
