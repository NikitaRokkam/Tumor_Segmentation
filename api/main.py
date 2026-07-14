import base64
import io
import logging
import os
import time
import uuid

import numpy as np
import torch
import torchvision.transforms.functional as TF
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tumor-seg-api")

MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/model_torchscript.pt")
API_KEY = os.environ.get("API_KEY")  # required -- no default, fails closed if unset
IMG_SIZE = 256
MASK_THRESHOLD = 0.5
MAX_FILE_SIZE_MB = 10
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}

app = FastAPI(title="Tumor Segmentation API", version="1.0.0")

_model = None  # loaded once at startup, kept in memory across requests


@app.on_event("startup")
def load_model():
    global _model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run src/export_model.py first.")
    _model = torch.jit.load(MODEL_PATH, map_location="cpu")
    _model.eval()
    logger.info("Model loaded from %s", MODEL_PATH)


def check_api_key(x_api_key: str = Header(...)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY not set")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class PredictionResponse(BaseModel):
    request_id: str
    tumor_detected: bool
    tumor_area_pixels: int
    tumor_area_percent: float
    mask_png_base64: str  # base64-encoded PNG of the binary mask, ready to render client-side
    inference_time_ms: float


def preprocess(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = TF.resize(image, [IMG_SIZE, IMG_SIZE])
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor.unsqueeze(0)  # add batch dim


def mask_to_base64_png(mask_array: np.ndarray) -> str:
    mask_image = Image.fromarray((mask_array * 255).astype(np.uint8))
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), _: None = Depends(check_api_key)):
    request_id = str(uuid.uuid4())
    t0 = time.time()

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    try:
        input_tensor = preprocess(image_bytes)
    except Exception:
        # Log the failure without logging the image bytes or filename (avoid PHI in logs)
        logger.warning("request_id=%s failed to decode uploaded image", request_id)
        raise HTTPException(status_code=400, detail="Could not decode image file")

    with torch.no_grad():
        logits = _model(input_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > MASK_THRESHOLD).float().squeeze().numpy()

    tumor_pixels = int(mask.sum())
    total_pixels = mask.size
    elapsed_ms = (time.time() - t0) * 1000

    logger.info("request_id=%s inference_time_ms=%.1f tumor_detected=%s",
                request_id, elapsed_ms, tumor_pixels > 0)

    return PredictionResponse(
        request_id=request_id,
        tumor_detected=tumor_pixels > 0,
        tumor_area_pixels=tumor_pixels,
        tumor_area_percent=round(100 * tumor_pixels / total_pixels, 2),
        mask_png_base64=mask_to_base64_png(mask),
        inference_time_ms=round(elapsed_ms, 1),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}
