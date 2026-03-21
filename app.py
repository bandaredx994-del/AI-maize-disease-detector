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

# 2. Model Download
FILE_ID = '17vu0TlrJBmoMvksYB62ufNA58WQRV6TZ'
url = f'https://drive.google.com/uc?id={FILE_ID}'
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

# 3. Class Names (Alphabetical to match Colab)
class_names = ['Common Rust', 'Gray Leaf Spot', 'Healthy', 'Northern Leaf Blight']

# 4. Scanner & File Uploader
st.subheader("📸 Scan a Maize Leaf")
option = st.radio("Choose how to provide a photo:", ("Use Camera Scanner", "Upload from Gallery"))

uploaded_file = None
if option == "Use Camera Scanner":
    uploaded_file = st.camera_input("Point camera at the leaf")
else:
    uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "png", "jpeg"])

# 5. Prediction & Results Logic (Only runs if a file is provided)
if uploaded_file is not None and model is not None:
    try:
        # Pre-processing
        image = Image.open(uploaded_file).convert('RGB') 
        st.image(image, caption='Target Leaf Photo', use_container_width=True)
        
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 6. Prediction Logic (Indented)
        with st.spinner('AI is analyzing...'):
            prediction = model.predict(img_array)
            result_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            st.write(f"DEBUG - Raw AI Scores: {prediction}")

        # 7. Results & Solutions (Indented)
        st.subheader(f"Diagnosis: **{class_names[result_index]}**")
        st.write(f"Confidence Score: **{confidence:.2f}%**")
        
        if class_names[result_index] == 'Healthy':
            st.success("✅ This plant is healthy! Continue standard care.")
        elif class_names[result_index] == 'Common Rust':
            st.warning("⚠️ **Common Rust Detected**")
            st.info("Solution: Apply fungicides early. Use rust-resistant seeds from SeedCo/Zamseed.")
        elif class_names[result_index] == 'Gray Leaf Spot':
            st.warning("⚠️ **Gray Leaf Spot Detected**")
            st.info("Solution: Rotate crops with beans. Practice deep tillage.")
        elif class_names[result_index] == 'Northern Leaf Blight':
            st.warning("⚠️ **Northern Leaf Blight Detected**")
            st.info("Solution: Remove/burn infected leaves. Increase plant spacing.")

        # 8. Interactive Section (Indented)
        st.divider()
        st.subheader("💬 Interact with the Maize Doctor")
        question = st.selectbox("Select a question to ask:", [
            "Select...", "Fall Armyworm?", "Stalk Borer?", "Prevention?", "Seeds?"
        ])

        if "Fall Armyworm" in question:
            st.error("🐛 Look for sawdust-like waste. Use Emamectin Benzoate.")
        elif "Stalk Borer" in question:
            st.error("🐛 Look for straight-line holes. Practice Push-Pull tech.")
        elif "Prevention" in question:
            st.info("Rotate with groundnuts or beans.")
        elif "Seeds" in question:
            st.info("Buy from SeedCo, Zamseed, or Pannar.")

        user_report = st.text_input("Report an outbreak:")
        if st.button("Submit Report"):
            st.success("Report logged!")

    except Exception as e:
        st.error(f"Error: {e}")

# 9. Sidebar QR Code (Always visible, NOT indented)
st.sidebar.markdown("---")
st.sidebar.write("📲 **Scan to open this app:**")
app_url = "https://ai-maize-disease-detector-mlx5aeghpfapd5vyn3bl2a.streamlit.app"
qr_api = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={app_url}"
st.sidebar.image(qr_api)
st.sidebar.info("Developed for the JETS Fair - Zambia")


