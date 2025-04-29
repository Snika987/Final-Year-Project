# Home.py
import streamlit as st

# Configure the page
st.set_page_config(page_title="Login", page_icon="🔒", layout="centered")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Redirect if already logged in
if st.session_state.logged_in:
    st.switch_page("pages/1_CRC_Detection.py")

# --- WELCOME ANIMATION ---
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExcTJuM2Rhc2NtZmtqcTlucjI4ODJzaG8yNWF3MGFmY2gwN3YxdDlmciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/mnQqFDhfdbjeEYJl0d/giphy.gif' width='300px'>
    </div>
    """, unsafe_allow_html=True
)

# Title
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🔒 Welcome to Medical Assistant</h1>", unsafe_allow_html=True)
st.write("### Please login to continue:")

# Login Form
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if username == "admin" and password == "abc":
        st.success("Login successful! 🎉")
        st.session_state.logged_in = True
        st.session_state.page = "CRC"
        st.rerun()

    else:
        st.error("Invalid credentials. Please try again.")
