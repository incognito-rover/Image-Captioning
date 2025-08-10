from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt

# Load pre-trained model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load a sample image
image_url = "https://images.unsplash.com/photo-1754152728457-902f155ebcae?q=80&w=2066&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
#image = Image.open("images/sample1.jpg").convert('RGB')
image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

# Process the image and generate caption
inputs = processor(images=image, return_tensors="pt").to(device)
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

# Show image with caption
plt.imshow(image)
plt.axis('off')
plt.title(caption, fontsize=14)
plt.show()

# Optional print
print("Generated Caption:", caption)
