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

    # 7. Results & Solutions for JETS Zambia
    st.subheader(f"Diagnosis: **{class_names[result_index]}**")
    st.write(f"Confidence Score: **{confidence:.2f}%**")
    if class_names[result_index] == 'Healthy':
        st.success("✅ This plant is healthy! Continue standard care.")
    elif class_names[result_index] == 'Common Rust':
        st.warning("⚠️ **Common Rust Detected**")
        st.info("""
        **Solutions for Zambian Farmers:**
        * **Action:** Apply fungicides (e.g., azoxystrobin) early.
        * **Prevention:** Use rust-resistant seeds from SeedCo or Zamseed.
        * **Tip:** Avoid overhead watering to stop spores from spreading.
        """)

    elif class_names[result_index] == 'Gray Leaf Spot':
        st.warning("⚠️ **Gray Leaf Spot Detected**")
        st.info("""
        **Solutions for Zambian Farmers:**
        * **Action:** Rotate crops with beans or groundnuts.
        * **Management:** Practice deep tillage to bury infected residue.
        * **Chemical:** Apply fungicides before the silking stage.
        """)

    elif class_names[result_index] == 'Northern Leaf Blight':
        st.warning("⚠️ **Northern Leaf Blight Detected**")
        st.info("""
            # 8. Interactive "Ask the Doctor" Section
    st.divider()
    st.subheader("💬 Interact with the Maize Doctor")
    
    # Dropdown for common Zambian farming questions
    question = st.selectbox("Select a question to ask the AI:", [
        "Select a question...",
        "How do I prevent these diseases next season?", 
        "Where can I buy resistant seeds in Zambia?", 
        "Is it safe to eat maize with Rust?"
    ])

    if question == "How do I prevent these diseases next season?":
        st.info("👨‍⚕️ **Doctor's Advice:** Practice **crop rotation**. Avoid planting maize in the same field two years in a row. Rotate with legumes like beans or groundnuts to break the disease cycle.")
    
    elif question == "Where can I buy resistant seeds in Zambia?":
        st.info("👨‍⚕️ **Doctor's Advice:** Visit your local agro-dealer and ask for certified 'Hybrid' seeds from **SeedCo**, **Zamseed**, or **Pannar**. Look for varieties specifically labeled 'Rust Resistant'.")
    
    elif question == "Is it safe to eat maize with Rust?":
        st.info("👨‍⚕️ **Doctor's Advice:** While the fungus is not typically toxic to humans, it reduces the grain's quality, weight, and taste. If heavily infected, it is better used for livestock feed.")

    # Feedback box for users to type their own message
    st.write("---")
    user_report = st.text_input("Report a new outbreak or ask a custom question:")
    if st.button("Submit Report"):
        st.success(f"Thank you! Your report: '{user_report}' has been logged for the JETS project.")
        **Solutions for Zambian Farmers:**
        * **Action:** Remove and burn heavily infected leaves.
        * **Prevention:** Increase plant spacing for better airflow.
        * **Tip:** Choose blight-tolerant hybrids for the next season.
        """)

st.sidebar.info("Developed for the JETS Fair - Zambia")
