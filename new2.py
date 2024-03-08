import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import numpy as np

# Function to load image
@st.cache(allow_output_mutation=True)
def load_image(image_file):
    img = Image.open(image_file)
    return img
# Load the pre-trained model

model_pneumonia = tf.keras.models.load_model('pneumoniadetection.h5')
model_tuberculosis = tf.keras.models.load_model('tuberculosis.h5')

def main():
    st.title("Pneumonia and tuberculosis Detection")
    st.write("Upload an image for Pneumonia and Tuberculosis detection")

    # Upload image for Pneumonia detection
    st.subheader("Pneumonia Detection")
    pneumonia_image_file = st.file_uploader("Upload X-ray image", type=['jpg', 'png'])
    if pneumonia_image_file is not None:
        img = load_image(pneumonia_image_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Resize and preprocess the image
        new_img = img.resize((128, 128))
        x = np.expand_dims(new_img, axis=0)
        input_image_scaled = x / 255.0

        # Make predictions
        input_prediction = model_pneumonia.predict(input_image_scaled)
        st.write("Prediction Probabilities:", input_prediction)
        input_pred_label = np.argmax(input_prediction)
        st.write("Predicted Label:", input_pred_label)

        # Display prediction
        if input_pred_label == 1:
            st.write("The person's X-ray in the image is suffering from pneumonia.")
        else:
            st.write("The person's X-ray in the image is normal, not suffering from pneumonia.")

    # Upload image for Tuberculosis detection
    st.subheader("Tuberculosis Detection")
    tuberculosis_image_file = st.file_uploader("Upload X-ray image1", type=['jpg', 'png'])
    if tuberculosis_image_file is not None:
        img = load_image(tuberculosis_image_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Resize and preprocess the image
        new_img = img.resize((128, 128))
        img_array = np.array(new_img)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:  # Check if image is RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        img_array = img_array.reshape((1, 128, 128, 1))  # Reshape to (1, 128, 128, 1)
        input_image_scaled = img_array / 255.0

        # Make predictions
        input_prediction = model_pneumonia.predict(input_image_scaled)
        st.write("Prediction Probabilities:", input_prediction)
        input_pred_label = np.argmax(input_prediction)
        st.write("Predicted Label:", input_pred_label)


        # Display prediction
        if input_pred_label == 1:
            st.write("The person's X-ray in the image is suffering from tuberculosis.")
        else:
            st.write("The person's X-ray in the image is normal, not suffering from tuberculosis.")

if __name__ == "__main__":
    main()
