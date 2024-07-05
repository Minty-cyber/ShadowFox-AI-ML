import streamlit as st
import torch
import torchvision
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io



weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()


transform = transforms.Compose([
    transforms.ToTensor()
])


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Function to visualize predictions
def visualize_predictions(image, predictions, threshold=0.5):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for idx, element in enumerate(predictions[0]['boxes']):
        score = predictions[0]['scores'][idx].item()
        if score > threshold:
            x1, y1, x2, y2 = element
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label_idx = predictions[0]['labels'][idx].item()
            label = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label}: {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Streamlit app
st.title("Object Detection with Faster R-CNN")
st.write("Upload an image to detect objects")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting objects...")

    # Transform the image
    transformed_image = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = model(transformed_image)

    # Visualize predictions
    result_image = visualize_predictions(image, predictions)
    
    # Convert result image to RGB format for display
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    st.image(result_image, caption='Processed Image.', use_column_width=True)

    # Print detected objects
    st.write("Detected objects:")
    for idx, element in enumerate(predictions[0]['boxes']):
        score = predictions[0]['scores'][idx].item()
        if score > 0.5:
            label_idx = predictions[0]['labels'][idx].item()
            label = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
            st.write(f"{label}: {score:.2f}")
