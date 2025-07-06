# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 00:12:15 2025

@author: dell
"""
import clip
import torch
from PIL import Image
import numpy as np

# Initialize CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Text Encoding Test
text_inputs = clip.tokenize(["a black wallet", "a red iPhone"]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
print(f"Text features shape: {text_features.shape}")  # Should be torch.Size([2, 512])

# Image Encoding Test
image = preprocess(Image.open("lost_wallet.jpg")).unsqueeze(0).to(device)
with torch.no_grad():
    image_features = model.encode_image(image)
print(f"Image features shape: {image_features.shape}")  # Should be torch.Size([1, 512])
