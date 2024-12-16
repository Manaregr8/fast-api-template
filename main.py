from typing import Optional

from fastapi import FastAPI
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
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # You can specify other languages too

# Initialize YOLO model for text region detection
model = YOLO("./best.pt")


# Function to detect text regions using YOLO model
def detect_text_regions(image_path):
    results = model(image_path)
    detections = results[0].boxes.xyxy.numpy()  # Bounding box coordinates
    return detections

# Function to crop detected text regions from the image
def crop_regions(image_path, detections, output_dir="cropped_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = cv2.imread(image_path)
    cropped_paths = []

    for i, (x1, y1, x2, y2) in enumerate(detections):
        cropped = image[int(y1):int(y2), int(x1):int(x2)]
        output_path = os.path.join(output_dir, f"region_{i}.jpg")
        cv2.imwrite(output_path, cropped)
        cropped_paths.append(output_path)

    return cropped_paths

# Function to apply sharpening to the image
def sharpen_image(image):
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # A basic sharpening filter
    return cv2.filter2D(image, -1, sharpening_kernel)

# Function to extract and standardize dates from OCR text
def extract_dates(text):
    text = re.sub(r'(\d{1,2}-[A-Za-z]{3}-\d{4})(\d{1,2}-[A-Za-z]{3}-\d{4})', r'\1 \2', text)
    text = re.sub(r'(\d{1,2}/\d{1,2}/\d{4})(\d{1,2}/\d{1,2}/\d{4})', r'\1 \2', text)

    dates = re.findall(date_pattern, text)
    standardized_dates = []

    month_abbr_to_num = {
        'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
        'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
    }

    for date in dates:
        date = date.strip()

        try:
            if '/' in date and len(date.split('/')) == 2 and len(date.split('/')[1]) == 2:
                month_str, year_str = date.split('/')
                if month_str.isdigit():
                    standardized_date = datetime.strptime(f"{month_str}/{year_str}", "%m/%y").strftime("%d/%m/%Y")
                else:
                    if month_str.upper() in month_abbr_to_num:
                        month_num = month_abbr_to_num[month_str.upper()]
                        standardized_date = datetime.strptime(f"{month_num}/{year_str}", "%m/%y").strftime("%d/%m/%Y")

            elif len(date.split('/')) == 3 and len(date.split('/')[2]) == 2:
                standardized_date = datetime.strptime(date, "%d/%m/%y").strftime("%d/%m/%Y")

            elif '-' in date and len(date.split('-')) == 3:
                standardized_date = datetime.strptime(date, "%d-%m-%Y").strftime("%d/%m/%Y")

            elif len(date.split('/')) == 3 and len(date.split('/')[2]) == 4:
                standardized_date = datetime.strptime(date, "%d/%m/%Y").strftime("%d/%m/%Y")

            elif len(date.split('/')) == 2 and len(date.split('/')[0]) == 2 and len(date.split('/')[1]) == 2:
                standardized_date = datetime.strptime(date + "/01", "%m/%y/%d").strftime("%d/%m/%Y")

            elif len(date.split('/')) == 2 and len(date.split('/')[0]) == 2 and len(date.split('/')[1]) == 4:
                standardized_date = datetime.strptime(date + "/01", "%m/%Y/%d").strftime("%d/%m/%Y")

            standardized_dates.append(standardized_date)
        except ValueError as e:
            print(f"Skipping invalid date: {date}. Error: {e}")

    return standardized_dates

# Function to process OCR and date comparison
def process_image(image_path):
    detections = detect_text_regions(image_path)
    all_recognized_texts = []

    if len(detections) > 0:
        cropped_paths = crop_regions(image_path, detections)
        for cropped_path in cropped_paths:
            image = cv2.imread(cropped_path)
            result = ocr.ocr(image, cls=True)
            text = " ".join([line[1][0] for line in result[0]])
            all_recognized_texts.append(text)
    else:
        image = cv2.imread(image_path)
        sharpened_image = sharpen_image(image)
        result = ocr.ocr(sharpened_image, cls=True)
        text = " ".join([line[1][0] for line in result[0]])
        all_recognized_texts.append(text)

    detected_dates = []
    for text in all_recognized_texts:
        detected_dates.extend(extract_dates(text))

    if detected_dates:
        dates_sorted = sorted(detected_dates, key=lambda x: datetime.strptime(x, "%d/%m/%Y"))
        mfg_date = dates_sorted[0]
        exp_date = dates_sorted[1] if len(dates_sorted) > 1 else None

        return {
            "manufacturing_date": mfg_date,
            "expiration_date": exp_date,
            "all_detected_dates": detected_dates
        }
    return {"error": "No valid dates detected"}
date_pattern = r"\b(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{2}|\d{2}/[A-Za-z]{3}/\d{4}|\d{2}/\d{4}|\b[A-Za-z]{3}/\d{2}|\d{2}/\d{2}|\d{2}-\d{2}-\d{4})\b"

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/process-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        result = process_image(temp_file)
        os.remove(temp_file)  # Clean up temp file

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)