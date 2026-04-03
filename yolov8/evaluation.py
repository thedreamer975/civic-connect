"""
yolov8_evaluation.py
Evaluate a pretrained YOLOv8 pothole detection model on a validation dataset,
printing key metrics without retraining.
"""

import os
from ultralytics import YOLO

# Paths and configs
MODEL_WEIGHTS = 'runs/segment/yolov8n-pothole-india/weights/best.pt'  # Adjust to your best model path
DATA_YAML = '/civic-models/dataset.yaml'  # Validation dataset config path

def main():
    # Load pretrained model weights
    model = YOLO(MODEL_WEIGHTS)
    
    # Run evaluation only (no training)
    metrics = model.val(data=DATA_YAML)

    # Print key metrics
    metrics = model.val(data=DATA_YAML)

    print('Evaluation metrics:')
    print(f"Mean Precision: {metrics.mp}")
    print(f"Mean Recall: {metrics.mr}")
    print(f"mAP50: {metrics.map50}")
    print(f"mAP50-95: {metrics.map}")

if __name__ == '__main__':
    main()
