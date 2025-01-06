# Import libraries
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
from PIL import Image

#########################################################
# Preprocessing images
def preprocess_image(image_path, target_size=(224, 224), remove_metadata=False):

    try:
        # Open image using Pillow
        with Image.open(image_path) as img:
            # Optionally remove metadata (fix for libpng warning)
            if remove_metadata and img.format == "PNG":
                # Remove PNG metadata by re-encoding
                img = img.convert("RGB")  
            # Convert to RGB format (handles grayscale and RGBA)
            img = img.convert("RGB")
            # Resize the image
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            # Convert to NumPy array and normalize pixel values
            img_array = np.array(img) / 255.0
            return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

##########################################################
# SSIM function
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
# Cosine similarity function
def compute_cosine_similarity(features1, features2):
    return 1 - cosine(features1, features2)

############################################################
# Common transform
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
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
    # print("\n### PyTorch Models ###")
    similarity_dict = {}

        # Preprocess both images (handle all formats, including PNG fixes)
    img1 = preprocess_image(img1_path, target_size=(224, 224), remove_metadata=True)
    img2 = preprocess_image(img2_path, target_size=(224, 224), remove_metadata=True)

    for name, model in dl_models.items():
        model.eval()
        with torch.no_grad():

            # Convert preprocessed images to tensors
            img1_tensor = transform(img1).unsqueeze(0)  # Add batch dimension
            img2_tensor = transform(img2).unsqueeze(0)

            # Extract features
            features1 = model(img1_tensor).squeeze().numpy()
            features2 = model(img2_tensor).squeeze().numpy()

            # Compute cosine similarity
            similarity = float(compute_cosine_similarity(features1, features2))
            # similarity_list.append(round(similarity,4))
            similarity_dict[name]= round(similarity,4)
            # print(f"{name} Cosine Similarity: {similarity:.4f}")

    return similarity_dict

def logo_similarity_make_decision (img1_path, valid_img, valid_img_path):

    # Call load_models to initialize both PyTorch models
    dl_models = load_models()
    final_decision = "no similarity"
    flag_decision = 0 
    for i in range(0, len(valid_img)):
        
        img2_path = f"{valid_img_path}{valid_img[i]}"
        # all_similarity= compute_similarity(img1_path, img2_path, dl_models, transform)
        print(valid_img[i])
        # print(f"all dl similarity are {all_similarity}")
        ssim_result_similarity = round(float(compute_ssim(img1_path, img2_path)),4)
        # # print(f"SSIM Score: {ssim_result_similarity:.4f}")
        # all_similarity["SSIM"]= ssim_result_similarity
        # print(f"all similarity are {all_similarity}")

        if ssim_result_similarity >=0.70:
            final_decision = f"similar to {valid_img[i]}"
            flag_decision =1
            model_name = "SSIM"
            print(f"{model_name} finds this image similar")
            return [final_decision,flag_decision, model_name, ssim_result_similarity]
        
        else:
            model_name = "SSIM"
            print(f"{model_name} is checked")
            dl_models_selected = {"MobileNet": mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)}
            dl_similarity = compute_similarity(img1_path, img2_path,dl_models_selected, transform)
            dl_similarity_value = dl_similarity["MobileNet"]
            if dl_similarity_value >0.7:
                final_decision = f"similar to {valid_img[i]}"
                flag_decision =1
                model_name = "MobileNet"
                print(f"{model_name} finds this image similar")
                return [final_decision,flag_decision, model_name, dl_similarity_value]

            else:
                model_name = "MobileNet"
                print(f"{model_name} is checked")
                dl_models_selected = {"EfficientNet_B0": efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)}
                dl_similarity = compute_similarity(img1_path, img2_path, dl_models_selected, transform)
                dl_similarity_value = dl_similarity["EfficientNet_B0"]
                if dl_similarity_value >0.7:
                    final_decision = f"similar to {valid_img[i]}"
                    flag_decision =1
                    model_name = "EfficientNet_B0"
                    print(f"{model_name} finds this image similar")
                    return [final_decision,flag_decision, model_name,dl_similarity_value]
                else:
                    model_name = "EfficientNet_B0"
                    print(f"{model_name} is checked")
                    dl_models_selected = {"VGG16": vgg16(weights=VGG16_Weights.DEFAULT)}
                    dl_similarity = compute_similarity(img1_path, img2_path, dl_models_selected, transform)
                    dl_similarity_value = dl_similarity["VGG16"]
                    if dl_similarity_value >0.8:
                        final_decision = f"similar to {valid_img[i]}"
                        flag_decision =1
                        model_name = "VGG16"
                        print(f"{model_name} finds this image similar")
                        return [final_decision,flag_decision, model_name,dl_similarity_value]
                    else:
                        model_name = "VGG16"
                        print(f"{model_name} is checked")
                        continue


    return ["There is no similarity" , 0 , "all models are checked", "None" ]
##########################################################################