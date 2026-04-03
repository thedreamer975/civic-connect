"""
yolov8_pipeline.py
End-to-end pipeline for pothole detection using Hugging Face YOLOv8 model.
Handles pretrained model download, fine-tuning, evaluation, and export.
"""
import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ---------------- Model Download ----------------
MODEL_REPO = 'keremberke/yolov8n-pothole-segmentation'
MODEL_FILENAME = 'best.pt'
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
print(f"Downloaded model weights to: {model_path}")

# ---------------- Dataset Config ----------------
DATA_YAML = '/civic-models/dataset.yaml'# Update path as needed
RESULTS_DIR = 'runs/segment/yolov8n-pothole-india'

# ---------------- Fine-tuning ----------------
model = YOLO(model_path)
model.train(
    data=DATA_YAML,
    epochs=20,
    batch=16,
    imgsz=640,
    project='runs/segment',
    name='yolov8n-pothole-india',
    exist_ok=True
)

# ---------------- Evaluation ----------------
metrics = model.val(data=DATA_YAML)

print('Evaluation metrics:')
print(f"Precision: {metrics.box.precision}")
print(f"Recall: {metrics.box.recall}")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

'''print('Evaluation metrics:')
    print(f"Mean Precision: {metrics.mp}")
    print(f"Mean Recall: {metrics.mr}")
    print(f"mAP50: {metrics.map50}")
    print(f"mAP50-95: {metrics.map}")'''

# ---------------- Export ----------------
export_dir = os.path.join(RESULTS_DIR, 'weights')
os.makedirs(export_dir, exist_ok=True)

# Export with dynamic input size enabled for flexibility
model.export(format='onnx', path=os.path.join(export_dir, 'model.onnx'), dynamic=True)
model.export(format='torchscript', path=os.path.join(export_dir, 'model.torchscript.pt'))

print('Model exported to ONNX and TorchScript formats.')
