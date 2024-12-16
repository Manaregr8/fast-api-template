from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
from ultralytics import YOLO

app = FastAPI()

# Load YOLO model
model = YOLO("./best.pt")

# Function to detect text regions using YOLO model
def detect_text_regions(image_path):
    results = model(image_path)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(float, box.xyxy.tolist()[0])
        detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return detections

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/process-image")
async def upload_image(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    try:
        # Save uploaded image to a temporary file
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        # Detect text regions
        result = detect_text_regions(temp_file)

        return JSONResponse(content={"detections": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
