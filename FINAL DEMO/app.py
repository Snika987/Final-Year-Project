import streamlit as st
import requests
from PIL import Image
import io
import base64
import warnings
import os

# Suppress warnings in Streamlit UI
warnings.filterwarnings("ignore")

# Suppress TensorFlow logging (deprecation warnings)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR

# Streamlit UI
st.title("Early Cancer Detection and Medical Chatbot")

# Section 1: Colorectal Cancer (CRC) Detection
st.subheader("Colorectal Cancer Detection")
st.write("Upload a histological image (.tif, .jpg, .png)")

crc_uploaded_file = st.file_uploader("Choose an image for CRC detection...", type=["tif", "tiff", "jpg", "png", "jpeg"], key="crc_uploader")

if crc_uploaded_file is not None:
    # Show uploaded image
    crc_image = Image.open(crc_uploaded_file)
    st.image(crc_image, caption="Uploaded CRC Image", use_container_width=True)

    # Convert image to bytes (ensure RGB format)
    if crc_image.mode != 'RGB':
        crc_image = crc_image.convert('RGB')
    img_bytes = io.BytesIO()
    crc_image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    if st.button("Predict CRC"):
        with st.spinner("Predicting CRC..."):
            # Send image to CRC FastAPI endpoint
            crc_url = "http://127.0.0.1:7000/predict_crc"
            files = {"file": ("image.png", img_bytes, "image/png")}
            try:
                response = requests.post(crc_url, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"**Prediction:** {result['prediction']}")
                    st.info(f"**Confidence:** {(result['confidence'] * 100):.2f}%")
                    # Display LIME explanation image
                    lime_image_data = result['explanation_image']
                    if lime_image_data.startswith("data:image/png;base64,"):
                        lime_image_bytes = base64.b64decode(lime_image_data.split(",")[1])
                        st.image(lime_image_bytes, caption="LIME Explanation (Highlighted Regions)", use_container_width=True)
                    else:
                        st.error("Invalid LIME image format")
                else:
                    st.error(f"Error: {response.json()['error']}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")

# Section 2: Kvasir Polyp Detection
st.subheader("Polyp Detection")
st.write("Upload an image (.jpg, .png)")

kvasir_uploaded_file = st.file_uploader("Choose an image for polyp detection...", type=["jpg", "png", "jpeg"], key="kvasir_uploader")

if kvasir_uploaded_file is not None:
    # Show uploaded image
    kvasir_image = Image.open(kvasir_uploaded_file)
    st.image(kvasir_image, caption="Uploaded Polyp Image", use_container_width=True)

    # Convert image to bytes
    kvasir_img_bytes = io.BytesIO()
    kvasir_image.save(kvasir_img_bytes, format="JPEG")
    kvasir_img_bytes = kvasir_img_bytes.getvalue()

    if st.button("Predict Polyp"):
        with st.spinner("Predicting..."):
            # Send image to Kvasir FastAPI endpoint
            kvasir_url = "http://127.0.0.1:8080/predict/"
            files = {"file": ("image.jpg", kvasir_img_bytes, "image/jpeg")}
            try:
                response = requests.post(kvasir_url, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"**Predicted Class:** {result['predicted_class']}")
                    st.info(f"**Polyp Status:** {result['polyp_status']}")
                else:
                    st.error(f"Error: {response.json().get('error', 'Could not get prediction')}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")

# Section 3: RAG Query
st.subheader("Ask a Medical Question")
query = st.text_input("Enter your medical question:")

if query:
    if st.button("Submit Question"):
        with st.spinner("Fetching answer..."):
            # Send query to RAG FastAPI endpoint
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