import streamlit as st
import requests
from PIL import Image
import io
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Page config
st.set_page_config(page_title="Polyp Detection & Medical Chat", page_icon="🩺", layout="centered")

# Check login
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Unauthorized access! Please login first.")
    st.stop()

# Check CRC prediction result
if "crc_result" not in st.session_state or st.session_state.crc_result is None:
    st.error("No CRC result found. Please complete CRC detection first.")
    st.stop()

# Title
st.title("🩺 Next Steps")

# Debug: Display crc_result for troubleshooting
st.write(f"Debug: CRC Result from previous page: {st.session_state.crc_result}")

# -------------- POLYP DETECTION -------------------
st.header("🩹 Polyp Detection")

# Check if cancer was detected
# Show file uploader only for non-cancerous results, no warning
if st.session_state.crc_result not in ["No Cancer", "Negative", "Cancer Absent"]:
    pass  # No warning or action when cancer is detected
else:
    kvasir_uploaded_file = st.file_uploader("Upload an image for polyp detection...", type=["jpg", "png", "jpeg"], key="kvasir_uploader")

    if kvasir_uploaded_file is not None:
        kvasir_image = Image.open(kvasir_uploaded_file)
        st.image(kvasir_image, caption="Uploaded Polyp Image", use_container_width=True)

        kvasir_img_bytes = io.BytesIO()
        kvasir_image.save(kvasir_img_bytes, format="JPEG")
        kvasir_img_bytes = kvasir_img_bytes.getvalue()

        if st.button("Predict Polyp"):
            with st.spinner("Predicting Polyp..."):
                kvasir_url = "http://127.0.0.1:8080/predict/"
                files = {"file": ("image.jpg", kvasir_img_bytes, "image/jpeg")}
                try:
                    response = requests.post(kvasir_url, files=files)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Predicted Class: {result['predicted_class']}")
                        st.info(f"Polyp Status: {result['polyp_status']}")
                    else:
                        st.error(f"Error: {response.json().get('error', 'Could not get prediction')}")
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")

# -------------- RAG Always Available -------------------
st.divider()

st.header("💬 Ask a Medical Question")

query = st.text_input("Enter your medical question:")

if query:
    if st.button("Submit Question"):
        with st.spinner("Fetching answer..."):
            rag_url = f"http://127.0.0.1:8000/query?query={query}"
            try:
                response = requests.get(rag_url)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Answer: {result['answer']}")
                else:
                    st.error(f"Error: {response.json().get('error', 'Could not get an answer')}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")