import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('/content/drive/MyDrive/trashnet/best_modelV2.h5', compile=False)

# Define class names
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("AI Model Test for Waste Classification")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image (similar to the first code)
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Get predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    score = predictions[0][predicted_class]
    
    # Display the results
    label = f"{class_names[predicted_class]}, Confidence: {100 * score:.2f}%"
    
    # Show the image with the predicted label
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(label)
    
    # Display using Matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.set_title(label)
    ax.axis("off")
    st.pyplot(fig)