from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
import faiss
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load models and index ONCE at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
yolo_model = YOLO("models/deepfashion2_yolov8s-seg.pt")
faiss_index = faiss.read_index("index/jersey_index.faiss")

loaded_data = np.load("index/jersey_metadata.npy", allow_pickle=True)
if isinstance(loaded_data, dict):
    index_to_path = {int(k): v for k, v in loaded_data.items()}
elif isinstance(loaded_data, np.ndarray):
    index_to_path = {i: str(item) for i, item in enumerate(loaded_data)}
else:
    index_to_path = {}

class FeaturesRequest(BaseModel):
    features: List[float] | List[List[float]]

@app.post("/dino")
async def dino_inference(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = dino_model(**inputs)
    features = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()[0]
    return {"features": features.tolist()}

@app.post("/faiss")
async def faiss_search(request: FeaturesRequest):
    features = request.features
    if isinstance(features[0], list):
        vector = np.array(features, dtype=np.float32)
    else:
        vector = np.array([features], dtype=np.float32)
    if vector.shape[1] != faiss_index.d:
        error_msg = f"Feature vector length {vector.shape[1]} does not match FAISS index dimension {faiss_index.d}"
        return {"error": error_msg}
    faiss.normalize_L2(vector)
    distances, indices = faiss_index.search(vector, 15)
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx in index_to_path:
            key = idx
            results.append({
                "rank": i + 1,
                "distance": float(distance),
                "file_path": index_to_path[key],
                "full_path": f"catalogue/{index_to_path[key]}"
            })
    return {"results": results}

@app.post("/yolo")
async def yolo_inference(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    results = yolo_model(image, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)[0]
    polygons = []
    if hasattr(results, 'masks') and results.masks is not None and hasattr(results.masks, 'xy'):
        for mask in results.masks.xy:
            polygons.append(mask.tolist())
    return {"polygons": polygons}