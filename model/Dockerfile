FROM python:3.11-slim

# System dependencies for Pillow + PyTorch
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install CPU-only PyTorch (much smaller than CUDA build ~200MB vs ~2GB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py predictor.py preprocessor.py ./
# Copy model files if they exist locally (they should be pushed to Git)
COPY font_model_scripted.pt class_index.json ./

# HF Spaces runs on port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
