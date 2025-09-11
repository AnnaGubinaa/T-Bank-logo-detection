from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from PIL import Image
import io
from ultralytics import YOLO


model = YOLO("weights/best.pt")
model.conf = 0.3

app = FastAPI(title="T-Bank Logo Detection API", version="1.0")

class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""
    x_min: int = Field(..., ge=0, description="Левая координата")
    y_min: int = Field(..., ge=0, description="Верхняя координата")
    x_max: int = Field(..., ge=0, description="Правая координата")
    y_max: int = Field(..., ge=0, description="Нижняя координата")

class Detection(BaseModel):
    """Результат детекции одного логотипа"""
    bbox: BoundingBox = Field(..., description="Результат детекции")

class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""
    detections: List[Detection] = Field(..., description="Список найденных логотипов")

class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")

@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, {
            "error": "Неподдерживаемый формат",
            "detail": "Ожидается изображение (JPEG, PNG, BMP, WEBP)"
        })

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        results = model(img, imgsz=640, max_det=10)

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) 
            detections.append(Detection(bbox=BoundingBox(
                x_min=x1,
                y_min=y1,
                x_max=x2,
                y_max=y2 
            )))

        return DetectionResponse(detections=detections)

    except Exception as e:
        raise HTTPException(500, {
            "error": "Ошибка обработки изображения",
            "detail": str(e)
        })