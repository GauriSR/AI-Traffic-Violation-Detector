import streamlit as st

st.title("🚦 AI Traffic Violation Detector")

st.write("Upload a traffic photo to detect violations.")

uploaded_file = st.file_uploader(
    "Choose a traffic image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Traffic Image",
        use_container_width=True
    )