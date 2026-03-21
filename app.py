import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# 1. Page Configuration
st.set_page_config(page_title="Zambian Maize Doctor", page_icon="🌽")

st.title("🌽 AI Maize Disease Detector")
st.markdown("""
### JETS Fair Project: Agriculture & AI
This app uses a **97% accurate** Deep Learning model to identify maize diseases in Zambia.
""")

# 2. FIXED: This section now downloads the model automatically
FILE_ID = '17vu0TlrJBmoMvksYB62ufNA58WQRV6TZ'
url = f'https://drive.google.com{FILE_ID}'
model_filename = 'maize_model.h5'

@st.cache_resource
def load_my_model():
    if not os.path.exists(model_filename):
        with st.spinner("Downloading AI Model from Google Drive..."):
            gdown.download(url, model_filename, quiet=False)
    try:
        return tf.keras.models.load_model(model_filename)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# 3. Class Names
class_names = ['Common Rust', 'Gray Leaf Spot', 'Northern Leaf Blight', 'Healthy']

# 4. File Uploader
uploaded_file = st.file_uploader("Upload a leaf photo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Target Leaf', use_container_width=True)
    
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner('AI is analyzing...'):
        prediction = model.predict(img_array)
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

    st.subheader(f"Diagnosis: **{class_names[result_index]}**")
    st.write(f"Confidence Score: **{confidence:.2f}%**")
    
    if class_names[result_index] == 'Healthy':
        st.success("✅ This plant is healthy!")
    else:
        st.warning(f"⚠️ Warning: {class_names[result_index]} detected.")

st.sidebar.info("Developed for the JETS Fair - Zambia")
