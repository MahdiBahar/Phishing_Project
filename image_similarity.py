# File: image_similarity_comparison.py

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, vgg16, efficientnet_b0
from torchvision.models import ResNet50_Weights, VGG16_Weights, EfficientNet_B0_Weights
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim

# Utility function for SSIM
def compute_ssim(image1_path, image2_path):
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to ensure consistent dimensions
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    
    similarity, _ = ssim(img1, img2, full=True)
    return similarity

# Utility functions for classical feature descriptors (SIFT)
def sift_similarity(image1_path, image2_path):
    sift = cv2.SIFT_create()
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # Match descriptors using FLANN
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return len(good_matches), len(kp1), len(kp2)

# Utility functions for deep learning models
def extract_features_dl(image_path, model, transform):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy()

def compute_cosine_similarity(features1, features2):
    return 1 - cosine(features1, features2)

def feature_extraction_and_similarity(image1_path, image2_path, model, transform):
    features1 = extract_features_dl(image1_path, model, transform)
    features2 = extract_features_dl(image2_path, model, transform)
    cosine_sim = compute_cosine_similarity(features1, features2)
    return cosine_sim

# Main comparison logic
def main(image1_path, image2_path):
    print("\n### Deep Learning Model Comparisons ###")

    # Load pre-trained models
    models = {
        "ResNet50": resnet50(weights=ResNet50_Weights.DEFAULT),
        "VGG16": vgg16( weights=VGG16_Weights.DEFAULT),
        "EfficientNet_B0": efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    }
    
    # Common transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Evaluate each model
    for name, model in models.items():
        model.fc = torch.nn.Identity()  # Remove classification layer for ResNet
        model.eval()
        similarity = feature_extraction_and_similarity(image1_path, image2_path, model, transform)
        print(f"{name} Cosine Similarity: {similarity:.4f}")

    print("\n### Structural Similarity Index (SSIM) ###")
    ssim_score = compute_ssim(image1_path, image2_path)
    print(f"SSIM Score: {ssim_score:.4f}")
    
    print("\n### Classical Feature Descriptor: SIFT ###")
    good_matches, kp1, kp2 = sift_similarity(image1_path, image2_path)
    print(f"Number of good matches: {good_matches}")
    print(f"Keypoints in Image 1: {kp1}, Image 2: {kp2}")
    print(f"SIFT Match Ratio: {good_matches / min(kp1, kp2):.4f}")

    print(f"comparison between {img1_name} and {img2_name}")

if __name__ == "__main__":
    # Replace with your image paths
    # img1_path = "/home/mahdi/phishing/bankmellat_pic5.png"
    img1_path = "/home/mahdi/phishing/bankmellat_pic3.jpeg"
    # img2_path = "/home/mahdi/phishing/banktejarat_pic8.png"
    # img2_path = "/home/mahdi/phishing/bankghavamin_pic7.jpg"
    img2_path = "/home/mahdi/phishing/bankmellat_pic4.png"
    img1_name = img1_path.split('/')[-1]
    img2_name = img2_path.split('/')[-1]
    main(img1_path, img2_path)
