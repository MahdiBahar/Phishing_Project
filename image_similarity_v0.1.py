## Installation part
# pip install opencv-python-headless scikit-image torch torchvision
## Run part
# File: image_similarity.py

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from scipy.spatial.distance import cosine

def compute_ssim(image1_path, image2_path):
    # Load images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize images to the same dimensions
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    
    # Compute SSIM
    similarity, _ = ssim(img1, img2, full=True)
    return similarity

def extract_features(image_path, model, transform):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply transformations
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Extract features using the pretrained model
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy()

def compute_cosine_similarity(features1, features2):
    return 1 - cosine(features1, features2)  # Convert distance to similarity

def main(image1_path, image2_path):
    print("Computing SSIM...")
    ssim_score = compute_ssim(image1_path, image2_path)
    print(f"SSIM Score: {ssim_score:.4f}")
    
    print("\nComputing feature-based similarity...")
    # Load pretrained ResNet50 model
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove the final classification layer
    model.eval()
    
    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Extract features
    features1 = extract_features(image1_path, model, transform)
    features2 = extract_features(image2_path, model, transform)
    
    # Compute cosine similarity
    cosine_sim = compute_cosine_similarity(features1, features2)
    print(f"Cosine Similarity: {cosine_sim:.4f}")

if __name__ == "__main__":
    # Replace with your image paths
    img1_path = "/home/mahdi/phishing/bankmellat_pic2.jpeg"
    img2_path = "/home/mahdi/phishing/bankmellat_pic5.png"
    
    main(img1_path, img2_path)
