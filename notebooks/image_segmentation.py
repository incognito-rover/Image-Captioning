import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import requests
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load image from URL
image_url = "https://images.unsplash.com/photo-1754152728457-902f155ebcae?q=80&w=2066&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# Transform image to tensor
image_tensor = F.to_tensor(image).unsqueeze(0)  # shape: [1, 3, H, W]

# Load pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
image_tensor = image_tensor.to(device)

# Predict
with torch.no_grad():
    outputs = model(image_tensor)

# Process outputs
threshold = 0.8  # confidence threshold
output = outputs[0]

# COCO category labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Draw results
image_np = np.array(image)
for i in range(len(output["scores"])):
    score = output["scores"][i].item()
    if score < threshold:
        continue

    box = output["boxes"][i].cpu().numpy().astype(int)
    label = COCO_INSTANCE_CATEGORY_NAMES[output["labels"][i]]
    mask = output["masks"][i, 0].cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)

    # Draw bounding box
    cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.putText(image_np, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # Apply mask overlay
    colored_mask = np.zeros_like(image_np, dtype=np.uint8)
    colored_mask[:, :, 1] = mask * 255  # Green mask
    image_np = cv2.addWeighted(image_np, 1.0, colored_mask, 0.5, 0)

# Show result
plt.figure(figsize=(10, 10))
plt.imshow(image_np)
plt.axis('off')
plt.title("Mask R-CNN Segmentation")
plt.show()
