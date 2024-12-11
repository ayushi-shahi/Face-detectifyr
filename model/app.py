import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Model Details
MODEL_NAME = "Face Detectifyr - Haar Cascade Classifier"
MODEL_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(MODEL_FILE)

# Define the function to detect faces
def detect_faces(image):
    try:
        # Convert the image to grayscale (Haar cascades work on grayscale images)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return image, len(faces)
    except Exception as e:
        st.error("Error in processing the image. Ensure the file is a valid image.")
        return image, 0

# Streamlit app
def main():
    st.title("Face Detection App")
    st.sidebar.title("Face Detection App")
    st.sidebar.info("Upload an image, and this app will detect faces using the Haar Cascade Model.")

    st.write(f"This app uses the **Face Detectifyr - Haar Cascade Classifier** for detecting faces in images.")

    # Instructions
    st.markdown("""
    ### How to Use:
    1. Upload an image in JPG, JPEG, or PNG format.
    2. The app will process the image and detect faces.
    3. Download the image with rectangles around detected faces if desired.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    # Example image button
    example_image_path = "assets/example.jpg"  # Ensure this file exists in the correct directory
    if st.button("Use Example Image"):
        uploaded_file = example_image_path

    if uploaded_file is not None:
        # Load the image
        if isinstance(uploaded_file, str):
            image = Image.open(uploaded_file)
        else:
            image = Image.open(uploaded_file)
        
        image_np = np.array(image)

        # Perform face detection
        result_image, face_count = detect_faces(image_np)

        # Display the original and result image
        st.image(image, caption="Uploaded Image", use_column_width=True, channels="RGB")
        st.image(result_image, caption=f"Detected Faces: {face_count}", use_column_width=True, channels="RGB")

        if face_count == 0:
            st.write("No faces detected in the uploaded image.")
        else:
            st.write(f"Number of faces detected: {face_count}")

        # Save and provide download option
        result_pil = Image.fromarray(result_image)
        result_pil.save("output.jpg")
        with open("output.jpg", "rb") as file:
            st.download_button(
                label="Download Image with Detected Faces",
                data=file,
                file_name="detected_faces.jpg",
                mime="image/jpeg"
            )

if __name__ == "__main__":
    main()
