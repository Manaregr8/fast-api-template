from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import re
from paddleocr import PaddleOCR
from datetime import datetime
from ultralytics import YOLO
import os

app = FastAPI()

# Initialize the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # You can specify other languages too

# Initialize YOLO model for text region detection
model = YOLO("./best.pt")


# Function to detect text regions using YOLO model
def detect_text_regions(image_path):
    results = model(image_path)
    detections = results[0].boxes.xyxy.numpy()  # Bounding box coordinates
    return detections

@app.get("/")
async def root():
    return {"message": "Hello World"}
# Updated regex pattern to match various date formats
date_pattern = r"\b(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{2}|\d{2}/[A-Za-z]{3}/\d{4}|\d{2}/\d{4}|\b[A-Za-z]{3}/\d{2}|\d{2}/\d{2}|\d{2}-\d{2}-\d{4})\b"

@app.post("/process-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        result = detect_text_regions(temp_file)
        os.remove(temp_file)  # Clean up temp file

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
