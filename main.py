from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.param_functions import Query  # Fixed import for Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np
import io
try:
    import requests
except ImportError:
    raise ImportError("The 'requests' library is required. Install it using 'pip install requests'")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Cho phép frontend React Vite truy cập
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load YOLO11n model (update path if needed)
model = YOLO("./models/y11sbest.pt")

@app.post("/predict_image_file/")
async def predict_image_file(file: UploadFile = File(...)):
    try:
        # Check if the uploaded file is an image
        if file.content_type.split("/")[0] != "image":
            raise HTTPException(status_code=400, detail="File is not an image.")

        # Read the image bytes and convert to OpenCV format
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        return run_inference(image)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class URLRequest(BaseModel):
    image_url: str

@app.post("/predict_url/")
async def predict_from_url(request: URLRequest):
    try:
        image_url = request.image_url
        # Download image from URL
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image from URL.")

        image_bytes = response.content
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image URL or file format.")

        return run_inference(image)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_inference(image):
    results = model(image)
    response = []

    for result in results:
        probs = result.probs.data.cpu().numpy()
        class_names = model.names

        # Get top 5 indices with highest probabilities
        top5_indices = np.argsort(probs)[-5:][::-1]

        # Prepare the response with class names and scores
        response = [
            {"class": class_names[int(idx)], "confidence": float(probs[idx])}
            for idx in top5_indices
        ]

    return JSONResponse(content={"top5_predictions": response})

@app.get("/")
def read_root():
    return {"message": "YOLO11n FastAPI is running. Use /predict/ to POST an image."}

# To run: uvicorn yolo_v8_fastapi:app --reload
