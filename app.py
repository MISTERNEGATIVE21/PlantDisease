import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = 'plant_disease_classification_model.h5'
IMAGE_SIZE = (128, 128)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Get class labels from training
IMAGE_DIR = 'images/images'
classes = [d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))]
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

# Prediction function
def predict(image):
    img = Image.open(image).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    class_idx = np.argmax(preds[0])
    class_name = idx_to_class[class_idx]
    confidence = float(preds[0][class_idx])
    return f"Prediction: {class_name} (Confidence: {confidence:.2f})"

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Upload Plant Image"),
    outputs=gr.Textbox(label="Result"),
    title="Plant Disease Detection",
    description="Upload a plant image to detect if it is healthy or has a disease."
)

if __name__ == "__main__":
    iface.launch()
