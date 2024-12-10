# File: image_similarity_comparison.py

import cv2
import numpy as np
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.transforms as transforms
from torchvision.models import (
    vgg16, efficientnet_b0, mobilenet_v2)
from torchvision.models import (
     VGG16_Weights, EfficientNet_B0_Weights, MobileNet_V2_Weights
    )
import matplotlib.pyplot as plt

##########################################################
# Utility function for SSIM
def compute_ssim(image1_path, image2_path):
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to ensure consistent dimensions
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    
    similarity, _ = ssim(img1, img2, full=True)
    return similarity

##############################################################

# Utility functions for classical feature descriptors (SIFT)
def sift_similarity(image1_path, image2_path):
    sift = cv2.SIFT_create()
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return 0, 0, 0, [], [], []
    
    # Match descriptors using FLANN
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return len(good_matches), len(kp1), len(kp2), kp1, kp2, good_matches


####################################################################
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

############################################################
## Preprocessing
# Common transform
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
# Preprocess image for PyTorch
def preprocess_dl_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)
    return img

############################################################
# Load pretrained models
def load_models():
    # PyTorch models
    pytorch_models = {
        "VGG16": vgg16(weights=VGG16_Weights.DEFAULT),
        "EfficientNet_B0": efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT),
        "MobileNet": mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT),
    }
    return pytorch_models


#  Extract features and compute cosine similarity
def compute_similarity(img1_path, img2_path, dl_models , transform):
    print("\n### PyTorch Models ###")
    for name, model in dl_models.items():
        model.eval()
        with torch.no_grad():
                        # Preprocess images
            img1 = preprocess_dl_image(img1_path)
            img2 = preprocess_dl_image(img2_path)
            # Extract features
            features1 = model(img1).squeeze().numpy()
            features2 = model(img2).squeeze().numpy()

            # Compute cosine similarity
            similarity = compute_cosine_similarity(features1, features2)
            print(f"{name} Cosine Similarity: {similarity:.4f}")


##########################################################################
if __name__ == "__main__":
    # Replace with your image paths
    # img1= 'bankmellat_pic3.jpeg'
# img2 = 'bankghavamin_pic7.jpg'
    img1= 'BM_LOGO-01.png' 
    # img2= 'banktejarat_pic8.png'
    # img1= 'cat.jpg'
    # img2= 'flower.jpg'
    # img1 = 'mellal.png'
    # img2 = 'bankmellat_pic6.jpeg'
    # img2= 'bankmellat_pic5.png'

    valid_img = ['BM_LOGO-00.png' , 'BM_LOGO-01.png' ,  'BM_LOGO-02.png' , 'BM_LOGO-03.png' , 'BM_LOGO-04.png', 'BM_LOGO-05.png']

    img1_path = f"/home/mahdi/Phishing_Project/Valid_images/{img1}"
    # show_image(img1_path, title="Original Image 1")

    # Call load_models to initialize both PyTorch models
    dl_models = load_models()
    for i in range(0, len(valid_img)):
        img2_path = f"/home/mahdi/Phishing_Project/Valid_images/{valid_img[i]}"
        # show_image(img2_path, title="Original Image 2")
        print(f"considered picture {img1} compare with {valid_img[i]}")
        print('Results:')
        compute_similarity(img1_path, img2_path, dl_models, transform)
        good_matches, kp1, kp2, keypoints1, keypoints2, matches = sift_similarity(img1_path, img2_path)
        print(f"SIFT Match Ratio: {good_matches / min(kp1, kp2):.4f}")
        ssim_score = compute_ssim(img1_path, img2_path)
        print(f"SSIM Score: {ssim_score:.4f}")
    