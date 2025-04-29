import streamlit as st
import requests
from PIL import Image
import io
import base64
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Page config
st.set_page_config(page_title="CRC Detection", page_icon="🧬", layout="centered")

# Session check
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Unauthorized access! Please login first.")
    st.stop()

if "crc_result" not in st.session_state:
    st.session_state.crc_result = None

# Title
st.title("🧬 Colorectal Cancer Detection")

st.write("Upload a histological image (.tif, .jpg, .png):")

crc_uploaded_file = st.file_uploader("Choose an image for CRC detection...", type=["tif", "tiff", "jpg", "png", "jpeg"], key="crc_uploader")

if crc_uploaded_file is not None:
    crc_image = Image.open(crc_uploaded_file)
    st.image(crc_image, caption="Uploaded CRC Image", use_container_width=True)

    if crc_image.mode != 'RGB':
        crc_image = crc_image.convert('RGB')
    img_bytes = io.BytesIO()
    crc_image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    if st.button("Predict CRC"):
        with st.spinner("Predicting CRC..."):
            crc_url = "http://127.0.0.1:7000/predict_crc"
            files = {"file": ("image.png", img_bytes, "image/png")}
            try:
                response = requests.post(crc_url, files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: {result['prediction']}")
                    st.info(f"Confidence: {(result['confidence'] * 100):.2f}%")
                    
                    st.session_state.crc_result = result['prediction']

                    lime_image_data = result['explanation_image']
                    if lime_image_data.startswith("data:image/png;base64,"):
                        lime_image_bytes = base64.b64decode(lime_image_data.split(",")[1])
                        st.image(lime_image_bytes, caption="LIME Explanation", use_container_width=True)
                    else:
                        st.warning("No valid explanation image.")
                    
                    st.session_state.ready_for_next = True  # Flag to move to next page

                else:
                    st.error(f"Error: {response.json()['error']}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")

# ➡️ Move to next page after prediction
if st.session_state.get("ready_for_next", False):
    if st.button("Proceed to Next Step ➡️"):
        st.switch_page(r"C:\Users\San_D\Desktop\Final App\pages\2_Polyp_and_Chatbot.py")  # This moves to page 2
