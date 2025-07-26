# 🦄 FastAPI Inference Server

A blazing-fast inference server for computer vision tasks using FastAPI, YOLO, DINOv2, and FAISS! 🚀

## Features
- **YOLOv8 Segmentation**: Detect and segment objects in images
- **DINOv2 Embeddings**: Extract powerful image features
- **FAISS Vector Search**: Find similar items using vector search
- **Easy REST API**: Simple endpoints for integration with any frontend

---

## 🛠️ Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-project-directory>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download/Place Model Files
- Place your YOLO model at `models/deepfashion2_yolov8s-seg.pt`
- Place your FAISS index and metadata at `index/jersey_index.faiss` and `index/jersey_metadata.npy`

### 4. Start the server
```bash
uvicorn inference_server:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔥 API Endpoints

### `POST /yolo`
- **Description:** Run YOLOv8 segmentation on an image
- **Request:** Multipart/form-data with an image file
- **Response:** JSON with detected polygons

### `POST /dino`
- **Description:** Extract DINOv2 features from an image
- **Request:** Multipart/form-data with an image file
- **Response:** JSON with feature vector

### `POST /faiss`
- **Description:** Search for similar items using FAISS
- **Request:** JSON with `features` (list of floats)
- **Response:** JSON with ranked search results

---

## 🧑‍💻 Example Usage

### YOLO Inference (Python)
```python
import requests

with open('your_image.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/yolo', files={'file': f})
print(response.json())
```

### DINOv2 Inference (Python)
```python
import requests

with open('your_image.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/dino', files={'file': f})
print(response.json())
```

### FAISS Search (Python)
```python
import requests
features = [0.1, 0.2, ...]  # Replace with your feature vector
response = requests.post('http://localhost:8000/faiss', json={'features': features})
print(response.json())
```

---

## 🌐 CORS
If using a frontend on a different port, make sure to enable CORS in `inference_server.py`:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📂 Project Structure
```
├── inference_server.py      # FastAPI app and endpoints
├── requirements.txt         # Python dependencies
├── models/
│   └── deepfashion2_yolov8s-seg.pt
├── index/
│   ├── jersey_index.faiss
│   └── jersey_metadata.npy
```

---

## 📝 License
MIT License 