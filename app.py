import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# 1. Page Configuration for JETS Zambia
st.set_page_config(page_title="Zambian Maize Doctor", page_icon="🌽")

st.title("🌽 AI Maize Disease Detector")
st.markdown("""
### JETS Fair Project: Agriculture & AI
This app uses a **97% accurate** Deep Learning model to identify maize diseases in Zambia.
""")

# 2. Model Download from Google Drive
FILE_ID = '17vu0TlrJBmoMvksYB62ufNA58WQRV6TZ'
url = f'https://drive.google.com/uc?id={FILEID}'
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

# 3. Class Names (Alphabetical to match Colab training)
class_names = ['Common Rust', 'Gray Leaf Spot', 'Healthy', 'Northern Leaf Blight']

# 4. Scanner & File Uploader
st.subheader("📸 Scan a Maize Leaf")
option = st.radio("Choose how to provide a photo:", ("Use Camera Scanner", "Upload from Gallery"))

uploaded_file = None
if option == "Use Camera Scanner":
    uploaded_file = st.camera_input("Point camera at the leaf")
else:
    uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "png", "jpeg"])

# 5. Process and Predict (Only if a file is provided)
if uploaded_file is not None and model is not None:
    try:
        # Pre-processing
        image = Image.open(uploaded_file).convert('RGB') 
        st.image(image, caption='Target Leaf Photo', use_container_width=True)
        
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 6. Prediction Logic
        with st.spinner('AI is analyzing...'):
            prediction = model.predict(img_array)
            # DEBUG line to show the judges the raw scores
            st.write(f"DEBUG - Raw AI Scores: {prediction}")
            
            result_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100

        # 7. Results & Solutions
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

        # 8. Smart Interactive "Ask the Doctor"
        st.divider()
        st.subheader("💬 Ask for a Solution")
        user_query = st.text_input("Type a question (e.g. 'how to fix rust' or 'when to plant')").lower()

        if user_query:
            if "solution" in user_query or "fix" in user_query or "treat" in user_query:
                st.info("👨‍⚕️ **Doctor's Solution:** Use fungicides like *Azoxystrobin* for spots. For worms, use *Emamectin Benzoate*. Always burn heavily infected leaves.")
            elif "fertilizer" in user_query or "manure" in user_query:
                st.info("👨‍⚕️ **Doctor's Solution:** Apply **D-Compound** at planting and **Urea** when the plant is knee-high.")
            elif "planting" in user_query or "when" in user_query:
                st.info("👨‍⚕️ **Doctor's Solution:** In Zambia, plant in early **November** after the first 25mm of rain.")
            elif "seeds" in user_query or "buy" in user_query:
                st.info("👨‍⚕️ **Doctor's Solution:** Buy certified hybrid seeds from **SeedCo, Zamseed, or Pannar**.")
            else:
                st.warning("👨‍⚕️ I'm still learning! Try asking about 'solutions', 'fertilizer', or 'planting'.")

        # 10. JETS Mini-Game: Maize Master Quiz
        st.divider()
        st.subheader("🎮 JETS Mini-Game: Maize Master Quiz")
        quiz_q = st.radio("Which Zambian crop is best for rotation to stop Maize diseases?", 
                         ["Tobacco", "Groundnuts (Bbalala)", "Cotton"])
        
        if st.button("Check Answer"):
            if quiz_q == "Groundnuts (Bbalala)":
                st.balloons()
                st.success("Correct! Legumes like groundnuts break the disease cycle. You are a Maize Master! 🏆")
            else:
                st.error("Not quite! Try rotating with legumes like Groundnuts or Beans. 🌽")

    except Exception as e:
        st.error(f"Error during analysis: {e}")

# 9. Sidebar QR Code (Always visible)
st.sidebar.markdown("---")
st.sidebar.write("📲 **Scan to open this app:**")
app_url = "https://ai-maize-disease-detector-mlx5aeghpfapd5vyn3bl2a.streamlit.app"
qr_api = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={app_url}"
st.sidebar.image(qr_api)
st.sidebar.info("Developed for the JETS Fair - Zambia")


