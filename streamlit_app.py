import streamlit as st
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as F
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import cv2
import io
import matplotlib.pyplot as plt

# Load models only once
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

@st.cache_resource
def load_segmentation_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

# COCO classes
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🖼️ Image Captioning + Segmentation")
st.write("Upload an image to generate a caption and segment objects.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load models
    caption_processor, caption_model, cap_device = load_caption_model()
    segment_model, seg_device = load_segmentation_model()

    # --- Image Captioning ---
    st.subheader("📝 Generated Caption")
    with st.spinner("Generating caption..."):
        inputs = caption_processor(images=image, return_tensors="pt").to(cap_device)
        output = caption_model.generate(**inputs)
        caption = caption_processor.decode(output[0], skip_special_tokens=True)
        st.success(caption)

    # --- Image Segmentation ---
    st.subheader("🔍 Image Segmentation")
    with st.spinner("Segmenting image..."):
        image_tensor = F.to_tensor(image).unsqueeze(0).to(seg_device)
        with torch.no_grad():
            preds = segment_model(image_tensor)[0]

        img_np = np.array(image)
        threshold = 0.8
        for i in range(len(preds["scores"])):
            if preds["scores"][i] < threshold:
                continue

            box = preds["boxes"][i].detach().cpu().numpy().astype(int)
            label = COCO_INSTANCE_CATEGORY_NAMES[preds["labels"][i]]
            mask = preds["masks"][i, 0].detach().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)

            # Draw box + label
            cv2.rectangle(img_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img_np, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Overlay mask
            color_mask = np.zeros_like(img_np)
            color_mask[:, :, 1] = mask * 255
            img_np = cv2.addWeighted(img_np, 1.0, color_mask, 0.4, 0)

        st.image(img_np, caption="Segmented Output", use_column_width=True)
