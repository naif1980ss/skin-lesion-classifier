# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from PIL import Image
import io, json, os
import tensorflow as tf

# === Config (can be overridden via env vars) ===
MODEL_DIR = os.getenv("MODEL_DIR", "models/skin_model.keras")
CLASS_FILE = os.getenv("CLASS_FILE", "models/class_names.json")

# === App ===
app = FastAPI(title="Skin Lesion Classifier")

# Serve everything in ./app; keep /predict working by mounting static at /static
app.mount("/static", StaticFiles(directory="app"), name="static")

# Serve the upload UI at "/" and "/index.html"
@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("app/index.html")

@app.get("/index.html", response_class=HTMLResponse)
def home_index():
    return FileResponse("app/index.html")

# === Load model & classes ===
# Expect a Keras 3 model file at models/skin_model.keras
if not os.path.exists(MODEL_DIR):
    raise RuntimeError(f"Model file not found at: {MODEL_DIR}")
if not os.path.exists(CLASS_FILE):
    raise RuntimeError(f"Class names file not found at: {CLASS_FILE}")

model = tf.keras.models.load_model(MODEL_DIR)
img_size = int(model.input_shape[1])

with open(CLASS_FILE, "r") as f:
    CLASS_NAMES = json.load(f)

# === Helpers ===
def preprocess_image(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image = image.resize((img_size, img_size))
    x = np.array(image, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)
    return x

# === Inference endpoint ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Please upload a JPEG or PNG image.")
    data = await file.read()
    x = preprocess_image(data)
    preds = model.predict(x)
    probs = preds[0].tolist()
    result = sorted(
        [{"class": CLASS_NAMES[i], "prob": float(p)} for i, p in enumerate(probs)],
        key=lambda d: d["prob"],
        reverse=True,
    )
    return {"predictions": result}

# === Entry point (optional) ===
if __name__ == "__main__":
    # Run: python app/main.py  (or use: uvicorn app.main:app --reload)
    uvicorn.run(app, host="0.0.0.0", port=8000)

