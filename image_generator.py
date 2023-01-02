import cv2
import streamlit as st
from transformers import pipeline

image_model = pipeline("text-generation", model="image-alpha-001")

def generate_image(prompt):
    image = image_model(prompt, max_length=1024, temperature=0.7)
    image = image[0]["image"]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

st.title("Image Generation")

prompt = st.text_input("Enter a text prompt")
if st.button("Generate"):
    image = generate_image(prompt)
    st.image(image)