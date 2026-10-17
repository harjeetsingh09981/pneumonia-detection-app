FROM python:3.11-slim

WORKDIR /app

# OpenCV runtime library (headless build still needs glib)
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first, so code changes don't trigger a full reinstall
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + model
COPY . .

# Defaults; override at run time with -e to switch to the champion model
ENV MODEL_PATH=model/cnn_model.keras \
    PREPROCESS=scratch \
    THRESHOLD=0.5

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
