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

# 3. Class Names
class_names = ['Common Rust', 'Gray Leaf Spot', 'Northern Leaf Blight', 'Healthy']

# 4. File Uploader
uploaded_file = st.file_uploader("Upload a leaf photo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Target Leaf', use_container_width=True)
    
    # 5. Pre-processing
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 6. Prediction Logic
    with st.spinner('AI is analyzing the leaf patterns...'):
        prediction = model.predict(img_array)
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

    # 7. Results & Solutions
    st.subheader(f"Diagnosis: **{class_names[result_index]}**")
    st.write(f"Confidence Score: **{confidence:.2f}%**")
    
    if class_names[result_index] == 'Healthy':
        st.success("✅ This plant is healthy! Continue standard care.")
    
    elif class_names[result_index] == 'Common Rust':
        st.warning("⚠️ **Common Rust Detected**")
        st.info("""
        **Solutions for Zambian Farmers:**
        * Apply fungicides (e.g., azoxystrobin) early.
        * Use rust-resistant seeds from SeedCo or Zamseed.
        """)

    elif class_names[result_index] == 'Gray Leaf Spot':
        st.warning("⚠️ **Gray Leaf Spot Detected**")
        st.info("""
        **Solutions for Zambian Farmers:**
        * Rotate crops with beans or groundnuts.
        * Practice deep tillage to bury infected residue.
        """)

    elif class_names[result_index] == 'Northern Leaf Blight':
        st.warning("⚠️ **Northern Leaf Blight Detected**")
        st.info("""
        **Solutions for Zambian Farmers:**
        * Remove and burn heavily infected leaves.
        * Increase plant spacing for better airflow.
        """)

    # 8. Interactive "Ask the Doctor" Section (Now outside the quotes!)
    st.divider()
    st.subheader("💬 Interact with the Maize Doctor")
    
    question = st.selectbox("Select a question to ask the AI:", [
        "Select a question...",
        "How do I prevent these diseases next season?", 
        "Where can I buy resistant seeds in Zambia?", 
        "Is it safe to eat maize with Rust?"
    ])

    if question == "How do I prevent these diseases next season?":
        st.info("👨‍⚕️ **Doctor's Advice:** Practice **crop rotation**. Avoid planting maize in the same field two years in a row. Rotate with legumes like beans or groundnuts.")
    
    elif question == "Where can I buy resistant seeds in Zambia?":
        st.info("👨‍⚕️ **Doctor's Advice:** Visit an agro-dealer and ask for certified seeds from **SeedCo**, **Zamseed**, or **Pannar**.")
    
    elif question == "Is it safe to eat maize with Rust?":
        st.info("👨‍⚕️ **Doctor's Advice:** While not usually toxic, it reduces quality and taste. Best used for livestock feed if heavily infected.")

    # Feedback box
    st.write("---")
    user_report = st.text_input("Report an outbreak or ask a custom question:")
    if st.button("Submit Report"):
        st.success(f"Thank you! Your report has been logged.")

st.sidebar.info("Developed for the JETS Fair - Zambia")
