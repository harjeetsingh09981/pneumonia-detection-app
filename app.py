"""Streamlit front end: upload a chest X-ray, get predicted class + probability."""
import os

import streamlit as st
import tensorflow as tf

from preprocessing import prepare

# --- Config via environment variables, so swapping in the champion model needs no code change ---
MODEL_PATH = os.getenv("MODEL_PATH", "model/cnn_model.keras")   # path to the saved .keras model
PREPROCESS = os.getenv("PREPROCESS", "scratch")                 # scratch | vgg16 | resnet50 (must match the model)
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))                # cut-off for flagging pneumonia

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="centered")


@st.cache_resource   # load once per container, not on every upload
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.caption("Decision-support prototype: a second opinion for radiologists, not a diagnosis.")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at `{MODEL_PATH}`. Copy the .keras file into the model/ folder.")
    st.stop()

model = load_model(MODEL_PATH)
uploaded = st.file_uploader("Upload a chest X-ray", type=["dcm", "png", "jpg", "jpeg"])

if uploaded is not None:
    display_img, batch = prepare(uploaded.getvalue(), uploaded.name, PREPROCESS)
    prob = float(model.predict(batch, verbose=0)[0][0])   # sigmoid output = probability of pneumonia
    label = "Pneumonia" if prob > THRESHOLD else "No Pneumonia"

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(display_img, caption="Preprocessed input (CLAHE, 224 × 224)", clamp=True)
    with col2:
        st.subheader("Prediction")
        (st.error if label == "Pneumonia" else st.success)(f"**{label}**")
        st.metric("Probability of pneumonia", f"{prob:.2%}")
        st.progress(prob)
        st.caption(f"Flagged as pneumonia above {THRESHOLD:.2f}. Model: `{os.path.basename(MODEL_PATH)}`")
