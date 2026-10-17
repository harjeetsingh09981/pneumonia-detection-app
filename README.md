# Pneumonia Detection App

Streamlit app for the PGP AI/ML Capstone 4. Upload a chest X-ray (DICOM, PNG or JPG) and get
the predicted class and probability of pneumonia.

## Run in GitHub Codespaces
1. Copy the trained model into `model/` (e.g. `model/cnn_model.keras`).
2. Build the image: `docker build -t pneumonia-app .`
3. Run it: `docker run -p 8501:8501 pneumonia-app`
4. Open the forwarded port 8501 from the **Ports** tab.

To use a different model, override the settings at run time:
`docker run -p 8501:8501 -e MODEL_PATH=model/<file>.keras -e PREPROCESS=vgg16 pneumonia-app`

*Decision-support prototype only; not for clinical diagnosis.*
