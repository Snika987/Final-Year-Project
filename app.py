import streamlit as st
from PIL import Image

# Hardcoded usernames and passwords
CREDENTIALS = {
    "admin": "admin123",
    "doctor": "doctor123",
    "user": "user123"
}

# Navigation between pages
def main():
    if "page" not in st.session_state:
        st.session_state.page = "Login"

    if st.session_state.page == "Login":
        login_page()
    elif st.session_state.page == "Upload":
        upload_page()
    elif st.session_state.page == "Prescriptive Analysis":
        prescriptive_analysis_page()

# Login Page
def login_page():
    st.title("Colorectal Cancer Detection - Login")
    st.markdown("### Please login to continue")

    username = st.text_input("Username", label_visibility="collapsed")
    password = st.text_input("Password", type="password", label_visibility="collapsed")

    if st.button("Login"):
        if username in CREDENTIALS and CREDENTIALS[username] == password:
            st.session_state.page = "Upload"
        else:
            st.error("Invalid credentials. Please try again.")

# Upload Page
def upload_page():
    st.title("Colorectal Cancer Detection")

    st.markdown("### Upload the scan image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        resized_image = image.resize((300, 300))  # Resize image to 300x300 pixels
        st.image(resized_image, caption="Uploaded Scan Image", use_container_width=True)
        # Simulate prediction (replace with your ML model prediction)
        prediction = st.radio("Prediction Result:", ["Cancerous", "Non-Cancerous", "Chances of Cancer"])

        if prediction == "Chances of Cancer":
            if st.button("Go to Prescriptive Analysis"):
                st.session_state.page = "Prescriptive Analysis"

# Prescriptive Analysis Page
def prescriptive_analysis_page():
    st.title("Prescriptive Analysis")

    st.markdown("### Ask your question")
    question = st.text_input("Question")
    if st.button("Submit Question"):
        # Simulated chatbot response (replace with actual chatbot logic)
        st.text_area("Chatbot Answer", "This is a simulated chatbot response.")

    if st.button("Back to Upload Page"):
        st.session_state.page = "Upload"

if __name__ == "__main__":
    st.set_page_config(page_title="Colorectal Cancer Detection")
    main()







