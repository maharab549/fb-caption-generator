# app.py

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
import torch

# Load BLIP model
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Load translation model
@st.cache_resource
def load_translator(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate(text, lang_code):
    models = {
        "bn": "Helsinki-NLP/opus-mt-en-bn",
        "hi": "Helsinki-NLP/opus-mt-en-hi"
    }
    tokenizer, model = load_translator(models[lang_code])
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# UI
st.set_page_config(page_title="Multilingual Facebook Caption Generator", layout="centered")
st.title("📸 Facebook Image Caption Generator")
st.markdown("Generate captions in **English**, **Bangla**, and **Hindi**")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        processor, model = load_blip()
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        en_caption = processor.decode(output[0], skip_special_tokens=True)
        bn_caption = translate(en_caption, "bn")
        hi_caption = translate(en_caption, "hi")

    st.markdown("### ✨ Captions")
    st.write(f"**🇬🇧 English:** {en_caption}")
    st.write(f"**🇧🇩 Bangla:** {bn_caption}")
    st.write(f"**🇮🇳 Hindi:** {hi_caption}")
