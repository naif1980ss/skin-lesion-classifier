# Real‑Time Skin Lesion Classifier (TensorFlow + FastAPI) — macOS (Apple Silicon)

End‑to‑end: data prep → transfer learning (EfficientNetV2B0) → Grad‑CAM → FastAPI deployment.

> **Disclaimer**: This is a technical demo, **not** a medical device.

---

## 0) Quick start on a new iMac (Apple Silicon)

### A) One‑time system prep
```bash
# 1) Xcode command line tools (compilers, git)
xcode-select --install

# 2) Homebrew (package manager). If you already have it, skip.
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 3) Miniforge (Conda for Apple Silicon)
brew install --cask miniforge
```
> After install, restart Terminal so `conda` is on PATH.

### B) Create your ML environment
```bash
# In the project folder:
conda create -n tf-mac python=3.11 -y
conda activate tf-mac

# For Apple Silicon GPU acceleration:
pip install --upgrade pip wheel setuptools
pip install -r requirements-apple-silicon.txt
```

### C) Verify TensorFlow + Metal
```bash
python scripts/test_tensorflow.py
```
You should see `mps` (Metal) listed as a device.

---

## 1) Data layout
Organize your dataset as:
```
data/
  train/
    benign/
    malignant/
  val/
    benign/
    malignant/
  test/
    benign/
    malignant/
```

---

## 2) Train the model
```bash
python train.py --epochs 10 --img-size 224 --batch-size 32
```
Artifacts:
- SavedModel → `models/skin_model/`
- Class names → `models/class_names.json`

---

## 3) Serve an API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Test:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@/path/to/image.jpg"
```

---

## 4) Grad‑CAM
```bash
python utils/gradcam.py --image /path/to/image.jpg --out heatmap.jpg
```

---

## 5) Docker (optional)
```bash
docker build -t skin-api .
docker run -p 8000:8000 skin-api
```

---

## 6) TFLite (optional)
```bash
python scripts/convert_savedmodel_to_tflite.py
```

---

## 7) Day‑by‑day refresh plan (10–14 days)

**Day 1–2:** macOS setup, env, sanity checks; skim TF/Keras basics.  
**Day 3–4:** Data ingestion + augmentation; baseline training run (5–10 epochs).  
**Day 5–6:** Fine‑tune backbone; try different img sizes/batch sizes; record metrics.  
**Day 7:** Add Grad‑CAM, sample visualizations; evaluate edge cases.  
**Day 8:** Wire up FastAPI; local prediction tests; log predictions.  
**Day 9:** Dockerize; run locally; simple cURL and HTTP tests.  
**Day 10–11:** Polish README; confusion matrix; precision/recall/F1 on `test`.  
**Day 12–13:** Optional cloud deploy (Render/Fly/EC2) and add tiny HTML upload UI.  
**Day 14:** Write a short LinkedIn/GitHub post; push repo public.

---

## 8) Notes for Intel Macs / Linux / Windows
Use `requirements-cpu.txt` instead of Apple Silicon. Create a venv or conda env, then:
```bash
pip install -r requirements-cpu.txt
```
