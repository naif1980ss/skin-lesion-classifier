# Convenience targets
.PHONY: help setup-mac train serve gradcam tflite

help:
	@echo "make setup-mac   - create conda env and install Apple Silicon TF"
	@echo "make train       - run training (edit EPOCHS/IMG/BATCH below)"
	@echo "make serve       - run FastAPI server"
	@echo "make gradcam     - run Grad-CAM on IMAGE=..."
	@echo "make tflite      - convert SavedModel to TFLite"

setup-mac:
	conda create -n tf-mac python=3.11 -y
	bash -lc "source activate tf-mac && pip install --upgrade pip wheel setuptools && pip install -r requirements-apple-silicon.txt"

EPOCHS?=10
IMG?=224
BATCH?=32

train:
	python train.py --epochs $(EPOCHS) --img-size $(IMG) --batch-size $(BATCH)

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

IMAGE?=sample.jpg
gradcam:
	python utils/gradcam.py --image $(IMAGE) --out heatmap.jpg

tflite:
	python scripts/convert_savedmodel_to_tflite.py
