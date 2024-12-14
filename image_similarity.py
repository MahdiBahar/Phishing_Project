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
from PIL import Image

#########################################################

def preprocess_image(image_path, target_size=(224, 224), remove_metadata=False):
    """
    Preprocesses an image to ensure compatibility for the deep learning pipeline.

    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Desired size (width, height) for resizing.
        remove_metadata (bool): If True, removes problematic metadata from PNG images.

    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array.
    """
    try:
        # Open image using Pillow
        with Image.open(image_path) as img:
            # Optionally remove metadata (fix for libpng warning)
            if remove_metadata and img.format == "PNG":
                img = img.convert("RGB")  # Remove PNG metadata by re-encoding
            # Convert to RGB format (handles grayscale and RGBA)
            img = img.convert("RGB")
            # Resize the image
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            # Convert to NumPy array and normalize pixel values
            img_array = np.array(img) / 255.0
            return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        raise





##########################################################
# Utility function for SSIM
# def compute_ssim(image1_path, image2_path):
#     img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
#     img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
#     # Resize to ensure consistent dimensions
#     img1 = cv2.resize(img1, (256, 256))
#     img2 = cv2.resize(img2, (256, 256))
    
#     similarity, _ = ssim(img1, img2, full=True)
#     return similarity

def compute_ssim(image1_path, image2_path):
    img1 = Image.open(image1_path).convert("L")
    img2 = Image.open(image2_path).convert("L")
    
    # Resize with updated resampling method
    img1 = img1.resize((256, 256), Image.Resampling.LANCZOS)
    img2 = img2.resize((256, 256), Image.Resampling.LANCZOS)
    
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    similarity, _ = ssim(img1_np, img2_np, full=True)
    return similarity
##############################################################

# Utility functions for classical feature descriptors (SIFT)
def sift_similarity(image1_path, image2_path):
    sift = cv2.SIFT_create()
    # img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Read and resize images using OpenCV
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Ensure consistent image size
    img1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (256, 256), interpolation=cv2.INTER_AREA)


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
# def extract_features_dl(image_path, model, transform):
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_tensor = transform(img).unsqueeze(0)
#     with torch.no_grad():
#         features = model(img_tensor)
#     return features.squeeze().numpy()


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
    
# # Preprocess image for PyTorch
# def preprocess_dl_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = transform(img).unsqueeze(0)
#     return img

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
    similarity_dict = {}

        # Preprocess both images (handle all formats, including PNG fixes)
    img1 = preprocess_image(img1_path, target_size=(224, 224), remove_metadata=True)
    img2 = preprocess_image(img2_path, target_size=(224, 224), remove_metadata=True)

    for name, model in dl_models.items():
        model.eval()
        with torch.no_grad():
            #             # Preprocess images
            # img1 = preprocess_dl_image(img1_path)
            # img2 = preprocess_dl_image(img2_path)
            # Convert preprocessed images to tensors
            img1_tensor = transform(img1).unsqueeze(0)  # Add batch dimension
            img2_tensor = transform(img2).unsqueeze(0)

            # Extract features
            features1 = model(img1_tensor).squeeze().numpy()
            features2 = model(img2_tensor).squeeze().numpy()
            # # Extract features
            # features1 = model(img1).squeeze().numpy()
            # features2 = model(img2).squeeze().numpy()

            # Compute cosine similarity
            similarity = float(compute_cosine_similarity(features1, features2))
            # similarity_list.append(round(similarity,4))
            similarity_dict[name]= round(similarity,4)
            print(f"{name} Cosine Similarity: {similarity:.4f}")

    return similarity_dict


def make_decision (img1_path, valid_img, transform):

    # Call load_models to initialize both PyTorch models
    dl_models = load_models()
    final_decision = "no similarity" 
    for i in range(0, len(valid_img)):
        count_vote = 0
        img2_path = f"/home/mahdi/Phishing_Project/Valid_images/{valid_img[i]}"
        print(f"considered picture {img1} compare with {valid_img[i]}")
        print('Results:')
        all_similarity= compute_similarity(img1_path, img2_path, dl_models, transform)
        print(f"all dl similarity are {all_similarity}")
        good_matches, kp1, kp2, keypoints1, keypoints2, matches = sift_similarity(img1_path, img2_path)
        SIFT_result_similarity = round((good_matches / min(kp1, kp2)),4)
        print(f"SIFT Match Ratio: {SIFT_result_similarity:.4f}")
        ssim_result_similarity = round(float(compute_ssim(img1_path, img2_path)),4)
        print(f"SSIM Score: {ssim_result_similarity:.4f}")

        all_similarity["SIFT"]= SIFT_result_similarity
        all_similarity["SSIM"]= ssim_result_similarity
        # final_result = [dl_similarity, SIFT_result_similarity,ssim_result_similarity ]
        print(f"all similarity are {all_similarity}")

        if all_similarity["VGG16"] >=0.25:
            count_vote+=1
        if all_similarity["EfficientNet_B0"]>=0.20:
            count_vote+=1
        if all_similarity["MobileNet"]>=0.20:
            count_vote+=1
        if all_similarity["SIFT"]>=0.20:
            count_vote+=1
        if all_similarity["SSIM"]>=0.20:
            count_vote+=1

        print(f"Vote count: {count_vote}")

          # Decision based on votes
        if count_vote > 3:
            print("Decision: Similar (Passed threshold)")
            final_decision = "similar"

    return final_decision
##########################################################################
if __name__ == "__main__":
    # Replace with your image paths
    img1= 'bankmellat_pic3.jpeg'
# img2 = 'bankghavamin_pic7.jpg'
    # img1= 'BM_LOGO-01.png' 
    # img2= 'banktejarat_pic8.png'
    # img1= 'cat.jpg'
    # img2= 'flower.jpg'
    # img1 = 'mellal.png'
    # img2 = 'bankmellat_pic6.jpeg'
    # img2= 'bankmellat_pic5.png'

    valid_img = ['BM_LOGO-00.png' , 'BM_LOGO-01.png' ,  'BM_LOGO-02.png' , 'BM_LOGO-03.png' , 'BM_LOGO-04.png', 'BM_LOGO-05.png']
    img1_path = f"/home/mahdi/Phishing_Project/images/{img1}"
    # img1_path = f"/home/mahdi/Phishing_Project/Valid_images/{img1}"
    # show_image(img1_path, title="Original Image 1")

    make_decision(img1_path, valid_img, transform)