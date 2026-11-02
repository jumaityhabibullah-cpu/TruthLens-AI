

# TruthLens AI

### AI-Powered Image Authenticity and Deepfake Detection System

## 1. Project Overview
TruthLens AI is a deep learning-based image classification system that identifies face images as REAL or AI-GENERATED (FAKE).

The project aims to raise awareness about the misuse of AI-generated images and deepfake technology, particularly the risks they pose to women and girls.

## 2. Dataset
Dataset: 140k Real and Fake Faces

Source: https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces

The dataset contains real face images and StyleGAN-generated fake face images.

## 3. Pretrained Model
Vision Transformer (ViT)

Model: google/vit-base-patch16-224-in21k

https://huggingface.co/google/vit-base-patch16-224-in21k

## 4. Fine-Tuned Model
TruthLens AI fine-tuned model:

https://huggingface.co/Jumaity/truthlens-ai

## 5. Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Streamlit
- Google Colab

## 6. Application Features
- Upload a face image.
- Analyze the uploaded image.
- Predict REAL or AI-GENERATED (FAKE).
- Display prediction confidence.

## 7. Model Performance
Test accuracy: 98.30%

Test precision: 98.43%

Test recall: 98.23%

Test F1-score: 98.33%

These results are from a 1,000-image test sample.

## 8. Limitations
The model was trained to distinguish real face images from StyleGAN-generated images. It cannot detect every type of deepfake or image manipulation. Its predictions are not proof of authenticity.

## 9. Developer
Jumaity
