
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(page_title="TruthLens AI", page_icon="🔍")

MODEL_ID = "Jumaity/truthlens-ai"

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model

st.title("🔍 TruthLens AI")
st.subheader("AI-Powered Image Authenticity Detection")

st.write("Upload a face image to check whether it is REAL or AI-GENERATED.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            processor, model = load_model()
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)[0]

            prediction = int(torch.argmax(probabilities))
            label = model.config.id2label[prediction]
            confidence = probabilities[prediction].item() * 100

        if label == "REAL":
            st.success("Prediction: REAL")
        else:
            st.warning("Prediction: AI-GENERATED (FAKE)")

        st.write(f"Confidence: {confidence:.2f}%")

        st.info(
            "This model was trained to distinguish real and StyleGAN-generated "
            "face images. It cannot detect every type of deepfake or manipulation. "
            "Its prediction is not proof of authenticity."
        )
