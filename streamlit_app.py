import streamlit as st
from streamlit_drawable_canvas import st_canvas
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, ImageOps
import torch
import numpy as np
import pandas as pd

st.set_page_config(page_title="Reconocedor MNIST (ViT)", layout="centered")
st.title('✏️ Dibuja un número (0‑9) y pulsa "Reconocer"')

@st.cache_resource
def cargar_modelo():
    model_id = "farleyknight/mnist-digit-classification-2022-09-04"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    return processor, model

processor, model = cargar_modelo()

canvas = st_canvas(
    fill_color="#000000", stroke_width=10, stroke_color="#FFFFFF",
    background_color="#000000", width=280, height=280,
    drawing_mode="freedraw", key="canvas"
)

if st.button("🔍 Reconocer número"):
    if canvas.image_data is not None:
        img = Image.fromarray(canvas.image_data[:, :, 0].astype("uint8"), mode="L")
        img = ImageOps.invert(img).resize((224, 224)).convert("RGB")

        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze()
            probs = torch.softmax(logits, dim=0).numpy()
            pred = int(np.argmax(probs))

        st.success(f"✅ Número reconocido: **{pred}**")
        df = pd.DataFrame({
            "Número": list(range(len(probs))),
            "Probabilidad": np.round(probs, 3)
        }).sort_values("Probabilidad", ascending=False)
        st.table(df)
    else:
        st.warning("⚠️ ¡Dibuja primero antes de reconocer!")
