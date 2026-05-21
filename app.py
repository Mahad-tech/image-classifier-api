from fastapi import FastAPI, UploadFile, File, HTTPException
from src.schemas import PredictionResponse
from src.model import load_model, predict_image
from PIL import Image
import io
import os

app = FastAPI(
    title="Image Classifier API",
    description="Classify flower images using fine-tuned EfficientNet-B0",
    version="1.0"
)

MODEL_PATH = "models/flowers_model.pth"

if os.path.exists(MODEL_PATH):
    print("Loading model...")
    model = load_model()
    print("Model loaded.")
else:
    model = None
    print("WARNING: No model found. Run python src/model.py first.")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="File must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read image")

    result = predict_image(model, image)
    return PredictionResponse(
        class_name=result["class_name"],
        confidence=result["confidence"]
    )
