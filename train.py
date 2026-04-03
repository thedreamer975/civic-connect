# train.py: Fine-tune Hugging Face YOLOv8 model on Indian pothole dataset
import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Download model weights from Hugging Face
MODEL_REPO = 'keremberke/yolov8n-pothole-segmentation'
MODEL_FILENAME = 'best.pt'
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
print(f"Downloaded model weights to: {model_path}")

# Dataset config
DATA_YAML = 'datasets/pothole-india/data.yaml'
RESULTS_DIR = 'runs/segment/yolov8n-pothole-india'

# Load YOLOv8 model
model = YOLO(model_path)

# Fine-tune on Indian dataset
model.train(
    data=DATA_YAML,
    epochs=50,
    batch=16,
    imgsz=640,
    project='runs/segment',
    name='yolov8n-pothole-india',
    exist_ok=True
)

# Evaluate on validation set
metrics = model.val(data=DATA_YAML)
print('Evaluation metrics:')
print(f"Precision: {metrics['metrics']['precision']}")
print(f"Recall: {metrics['metrics']['recall']}")
print(f"mAP50: {metrics['metrics']['mAP50']}")
print(f"mAP50-95: {metrics['metrics']['mAP50-95']}")

# Export model to ONNX and TorchScript
export_dir = os.path.join(RESULTS_DIR, 'weights')
os.makedirs(export_dir, exist_ok=True)
model.export(format='onnx', path=os.path.join(export_dir, 'model.onnx'))
model.export(format='torchscript', path=os.path.join(export_dir, 'model.torchscript.pt'))
print('Model exported to ONNX and TorchScript formats.')
