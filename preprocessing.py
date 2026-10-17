"""Image preprocessing for the pneumonia app.
Must mirror the training notebook exactly (CLAHE -> 224x224 resize -> 3 channels -> model-specific
normalization), otherwise predictions silently degrade."""
import io

import cv2
import numpy as np
import pydicom
from PIL import Image

IMG_SIZE = 224   # same input size used for every model in the notebook


def load_grayscale(file_bytes: bytes, filename: str) -> np.ndarray:
    """Read a DICOM or PNG/JPG upload into a 2-D uint8 grayscale array."""
    if filename.lower().endswith(".dcm"):
        arr = pydicom.dcmread(io.BytesIO(file_bytes)).pixel_array
    else:
        arr = np.array(Image.open(io.BytesIO(file_bytes)).convert("L"))   # "L" = 8-bit grayscale
    return arr.astype("uint8")


def apply_clahe(pixel_array: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Same CLAHE settings as notebook section 5.3."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(pixel_array.astype("uint8"))


def normalize(rgb: np.ndarray, mode: str) -> np.ndarray:
    """Model-specific normalization, matching the generator's preprocess_fn in the notebook."""
    rgb = rgb.astype("float32")
    if mode == "scratch":                       # Model 1: simple 0-1 scaling
        return rgb / 255.0
    if mode == "vgg16":                         # VGG-16 models: ImageNet zero-centring
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input(rgb)
    if mode == "resnet50":                      # ResNet50 models
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input(rgb)
    raise ValueError(f"Unknown PREPROCESS mode: {mode}")


def prepare(file_bytes: bytes, filename: str, mode: str):
    """Full pipeline. Returns (display image, model-ready batch of shape 1x224x224x3)."""
    gray = load_grayscale(file_bytes, filename)
    enhanced = apply_clahe(gray)
    resized = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
    rgb = np.stack([resized] * 3, axis=-1)      # 1 channel -> 3, as in training
    return resized, normalize(rgb, mode)[np.newaxis, ...]
