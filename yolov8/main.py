# main.py: FastAPI app for pothole detection inference
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

# Load fine-tuned model weights
MODEL_PATH = 'runs/segment/yolov8n-pothole-india/weights/best.pt'
model = YOLO(MODEL_PATH)

@app.post('/infer/')
async def infer(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    img_array = np.array(image)
    results = model(img_array)
    predictions = []
    for r in results:
        for box, score, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            predictions.append({
                'class': r.names[int(cls)],
                'confidence': float(score),
                'bbox': [float(x) for x in box]
            })
    return JSONResponse(content={'predictions': predictions})
