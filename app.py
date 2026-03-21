import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration for JETS Presentation
st.set_page_config(page_title="Zambian Maize Doctor", page_icon="🌽")

st.title("🌽 AI Maize Disease Detector")
st.markdown("""
### JETS Fair Project: Agriculture & AI
This app uses a **97% accurate** Deep Learning model to identify maize diseases in Zambia.
""")

# 2. Load the 'Brain' (The model you downloaded from Colab)
@st.cache_resource
def load_my_model():
    try:
        return tf.keras.models.load_model('maize_model.h5')
    except Exception as e:
        st.error(f"Error: Could not find 'maize_model.h5' in this folder. {e}")
        return None

model = load_my_model()

# 3. Define the Class Names (Must match your Colab folder order)
class_names = ['Common Rust', 'Gray Leaf Spot', 'Northern Leaf Blight', 'Healthy']

# 4. File Uploader
uploaded_file = st.file_uploader("Upload a leaf photo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Target Leaf', use_container_width=True)
    
    # 5. Pre-processing (Preparing the photo for the AI)
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 6. Prediction Logic
    with st.spinner('AI is analyzing the leaf patterns...'):
        prediction = model.predict(img_array)
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

    # 7. Results Display
    st.subheader(f"Diagnosis: **{class_names[result_index]}**")
    st.write(f"Confidence Score: **{confidence:.2f}%**")
    
    if class_names[result_index] == 'Healthy':
        st.success("✅ This plant is healthy! Continue standard care.")
    else:
        st.warning(f"⚠️ Warning: {class_names[result_index]} detected.")
        st.info("Recommendation: Consult your local Agricultural Extension Officer for treatment options.")

st.sidebar.info("Developed for the JETS Fair - Zambia")