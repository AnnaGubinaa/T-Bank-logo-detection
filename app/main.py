from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from PIL import Image
import io
from ultralytics import YOLO

model = YOLO("weights/best.pt")  # путь относительно корня проекта

app = FastAPI(title="T-Bank Logo Detection API", version="1.0")

class BoundingBox(BaseModel):
    x_min: int = Field(..., ge=0, description="Левая координата")
    y_min: int = Field(..., ge=0, description="Верхняя координата")
    x_max: int = Field(..., ge=0, description="Правая координата")
    y_max: int = Field(..., ge=0, description="Нижняя координата")

class Detection(BaseModel):
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    detections: List[Detection]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении
    Поддерживаемые форматы: JPEG, PNG, BMP, WEBP
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, {
            "error": "Неподдерживаемый формат",
            "detail": "Ожидается изображение (JPEG, PNG, BMP, WEBP)"
        })

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        results = model(img)

        detections = []
        for det in results.xyxy[0]:  
            x1, y1, x2, y2, conf, cls = det.tolist()
            detections.append(Detection(bbox=BoundingBox(
                x_min=int(x1), y_min=int(y1),
                x_max=int(x2), y_max=int(y2)
            )))

        return DetectionResponse(detections=detections)

    except Exception as e:
        raise HTTPException(500, {
            "error": "Ошибка обработки изображения",
            "detail": str(e)
        })