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

          # 8. Smart Interactive "Ask the Doctor"
    st.divider()
    st.subheader("💬 Ask the Doctor for a Solution")
    
    # Let the user type their own question!
    user_query = st.text_input("Type your question here (e.g., 'How do I fix rust?' or 'When to plant?')").lower()

    if user_query:
        if "solution" in user_query or "fix" in user_query or "treat" in user_query:
            st.info("👨‍⚕️ **Doctor's Solution:** If you see spots, use fungicides like *Azoxystrobin*. If you see worms, use *Emamectin Benzoate*. Always remove heavily infected leaves and burn them.")
        
        elif "fertilizer" in user_query or "manure" in user_query:
            st.info("👨‍⚕️ **Doctor's Solution:** Maize needs Nitrogen! Apply **D-Compound** at planting and **Urea** top-dressing when the plant is knee-high (V6 stage).")
            
        elif "planting" in user_query or "when" in user_query:
            st.info("👨‍⚕️ **Doctor's Solution:** In Zambia, aim for the first week of **November** or after the first 25mm of rain. Early planting usually means higher yields!")
            
        elif "seeds" in user_query or "buy" in user_query:
            st.info("👨‍⚕️ **Doctor's Solution:** Buy certified hybrid seeds from **SeedCo, Zamseed, or Pannar**. They have better resistance to the diseases this AI just scanned!")
            
        else:
            st.warning("👨‍⚕️ I'm still learning! Try asking about 'solutions', 'fertilizer', 'planting', or 'seeds'.")


# 9. Sidebar QR Code (Always visible, NOT indented)
st.sidebar.markdown("---")
st.sidebar.write("📲 **Scan to open this app:**")
app_url = "https://ai-maize-disease-detector-mlx5aeghpfapd5vyn3bl2a.streamlit.app"
qr_api = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={app_url}"
st.sidebar.image(qr_api)
st.sidebar.info("Developed for the JETS Fair - Zambia")

        # 10. JETS Mini-Game: Maize Master Quiz
        st.divider()
        st.subheader("🎮 JETS Mini-Game: Maize Master Quiz")
        
        # List of different Zambian farming questions
        if 'quiz_index' not in st.session_state:
            st.session_state.quiz_index = np.random.randint(0, 3)

        questions = [
            {
                "q": "Which crop is best for rotation to stop Maize diseases?",
                "options": ["Tobacco", "Groundnuts (Bbalala)", "Cotton"],
                "a": "Groundnuts (Bbalala)",
                "note": "Legumes like groundnuts add nitrogen back to the soil!"
            },
            {
                "q": "What is the main sign of Fall Armyworm damage?",
                "options": ["Yellow spots", "Window-pane holes", "Purple edges"],
                "a": "Window-pane holes",
                "note": "They eat through the leaf layers leaving a clear 'window'!"
            },
            {
                "q": "Which Zambian month is usually best for early maize planting?",
                "options": ["August", "November", "March"],
                "a": "November",
                "note": "Planting with the first rains in November ensures the best growth!"
            }
        ]

        current = questions[st.session_state.quiz_index]
        user_choice = st.radio(current["q"], current["options"])

        if st.button("Check Answer"):
            if user_choice == current["a"]:
                st.balloons()
                st.success(f"Correct! 🏆 {current['note']}")
                if st.button("Next Question"):
                    st.session_state.quiz_index = (st.session_state.quiz_index + 1) % 3
                    st.rerun()
            else:
                st.error(f"Not quite! {current['note']} 🌽")

