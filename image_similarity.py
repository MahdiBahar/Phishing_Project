# Import libraries
import numpy as np
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import (
    vgg16, efficientnet_b0, mobilenet_v2)
from torchvision.models import (
     VGG16_Weights, EfficientNet_B0_Weights, MobileNet_V2_Weights
    )
from PIL import Image , UnidentifiedImageError
import cairosvg
# import logging 

#########################################################
# convert svg to png
def check_and_convert_svgtopng(image_path):
    try:
        if image_path.lower().endswith(".svg"):
            print(f"Converting SVG to PNG: {image_path}")
            # Convert SVG to PNG in memory
            png_path = image_path.replace(".svg", ".png")
            cairosvg.svg2png(url=image_path, write_to=png_path)
            image_path = png_path  # Update path to use the converted PNG
        else:
            pass

        return image_path  
    
    except UnidentifiedImageError:
        print(f"Invalid image file: {image_path}")
    except Exception as e:
        print(f"Unexpected error processing {image_path}: {e}")
    return None

# Preprocessing images
def preprocess_image(image_path, target_size=(224, 224), remove_metadata=False):

    try:
        # Open image using Pillow
        with Image.open(image_path) as img:
            
             #Check if the image has a palette and transparency
            if img.mode == "P" and "transparency" in img.info:
                # print("Converting palette-based image to RGBA.")
                img = img.convert("RGBA")
            # Optionally remove metadata (fix for libpng warning)
            elif remove_metadata and img.format == "PNG":
                # Remove PNG metadata by re-encoding
                img = img.convert("RGB")  
            
            
            # Convert to RGB format (handles grayscale and RGBA)
            img = img.convert("RGB")
            # Resize the image
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            # Convert to NumPy array and normalize pixel values
            img_array = np.array(img) / 255.0
            return img_array
    except FileNotFoundError:
        print(f"File not found: {image_path}")
    except UnidentifiedImageError:
        print(f"Invalid image file: {image_path}")
    except Exception as e:
        print(f"Unexpected error processing {image_path}: {e}")
    return None

##########################################################
# SSIM function
def compute_ssim(image1_path, image2_path):
    try:
        # # Open images in grayscale
        # img1 = Image.open(image1_path).convert("L")
        # img2 = Image.open(image2_path).convert("L")
        
        # # Resize with updated resampling method
        # img1 = img1.resize((256, 256), Image.Resampling.LANCZOS)
        # img2 = img2.resize((256, 256), Image.Resampling.LANCZOS)
        
        # img1_np = np.array(img1)
        # img2_np = np.array(img2)

        # similarity, _ = ssim(img1_np, img2_np, full=True)
        # return similarity

                # Preprocess both images (handles .svg to .png conversion)
        img1 = preprocess_image(image1_path, target_size=(256, 256))
        img2 = preprocess_image(image2_path, target_size=(256, 256))

        if img1 is None or img2 is None:
            print("Failed to preprocess one or both images.")
            return -1.0

        # Convert to grayscale
        img1_gray = np.mean(img1, axis=-1)  # Convert RGB to grayscale
        img2_gray = np.mean(img2, axis=-1)

        # Compute SSIM
        similarity, _ = ssim(img1_gray, img2_gray, full=True, data_range=1.0)
        return round(similarity, 4)


    # except FileNotFoundError:
    #     print(f"File not found: {image1_path} or {image2_path}")
    # except UnidentifiedImageError:
    #     print(f"Invalid image file: {image1_path}")
    except Exception as e:
        print(f"Unexpected error computing SSIM: {e}")
    return -1.0
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
    """
    Load models including the fine-tuned EfficientNet model.
    """
    # Path to the fine-tuned model
    model_path = "/home/mahdi/Phishing_Project/trained_model/efficientnet_finetuned_1.pth"

    # Initialize EfficientNet model
    efficientnet_finetuned = models.efficientnet_b0(weights=None)
    num_classes = 2  # Replace with the number of classes in your fine-tuned model
    efficientnet_finetuned.classifier[1] = torch.nn.Linear(
        efficientnet_finetuned.classifier[1].in_features, num_classes
    )

    # Load fine-tuned weights
    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        efficientnet_finetuned.load_state_dict(state_dict, strict=True)  # Ensure strict=True to match exact layers
    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}")
        print("Attempting to load with strict=False...")
        efficientnet_finetuned.load_state_dict(state_dict, strict=False)

    efficientnet_finetuned.eval()  # Set to evaluation mode

    # Other models
    pytorch_models = {
        "VGG16": vgg16(weights=VGG16_Weights.DEFAULT),
        "EfficientNet_B0_FineTuned": efficientnet_finetuned,
        # "EfficientNet_B0": efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT),
        "MobileNet": mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    }
    return pytorch_models



# def load_models():

#     # Load fine-tuned EfficientNet model
#     model_path = "/home/mahdi/Phishing_Project/trained_model/efficientnet_finetuned.pth"
#     efficientnet_finetuned = models.efficientnet_b0(weights=None)  # Initialize without pretrained weights
#     efficientnet_finetuned.classifier[1] = torch.nn.Identity()  # For feature extraction
#     efficientnet_finetuned.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
#     efficientnet_finetuned.eval()  # Set to evaluation mode

#     # PyTorch models
#     pytorch_models = {
#         "VGG16": vgg16(weights=VGG16_Weights.DEFAULT),
#         # "EfficientNet_B0": efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT),
#         "EfficientNet_B0_FineTuned": efficientnet_finetuned,  # Use the fine-tuned model
#         "MobileNet": mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT),
#     }
#     return pytorch_models


#  Extract features and compute cosine similarity
def compute_similarity(img1_path, img2_path, dl_models , transform):
    # print("\n### PyTorch Models ###")
    similarity_dict = {}

        # Preprocess both images (handle all formats, including PNG fixes)
    img1 = preprocess_image(img1_path, target_size=(224, 224), remove_metadata=True)
    img2 = preprocess_image(img2_path, target_size=(224, 224), remove_metadata=True)
    
    #check the image is invalid or not
    if img1 is None or img2 is None:
        # print(f"Skipping comparison due to invalid images: {img1_path}, {img2_path}")
        print(f"Skipping comparison due to invalid images")
        return {"error": "Invalid image(s)"}

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
    img1_path = check_and_convert_svgtopng(img1_path)
    if img1_path is None:
        return ["An error is occured" , 0 , "None" , "None"]
    else: 
        for i in range(0, len(valid_img)):
            
            img2_path = f"{valid_img_path}{valid_img[i]}"
            # all_similarity= compute_similarity(img1_path, img2_path, dl_models, transform)
            print(valid_img[i])
            # print(f"all dl similarity are {all_similarity}")
            ssim_result_similarity = round(float(compute_ssim(img1_path, img2_path)),4)
            # # print(f"SSIM Score: {ssim_result_similarity:.4f}")
            # all_similarity["SSIM"]= ssim_result_similarity
            # print(f"all similarity are {all_similarity}")
            if ssim_result_similarity == -1:
                return ["An error is occured" , 0 , "None" , "None"]
            else:
                if ssim_result_similarity >0.8:
                    final_decision = f"similar to {valid_img[i]}"
                    flag_decision =1
                    model_name = "SSIM"
                    print(f"{model_name} finds this image similar")
                    return [final_decision,flag_decision, model_name, ssim_result_similarity]
                
                else:

                    dl_models_selected = {"EfficientNet_FineTuned": dl_models["EfficientNet_B0_FineTuned"]}
                    dl_similarity = compute_similarity(img1_path, img2_path, dl_models_selected, transform)
                    if "error" in dl_similarity:
                        return ["An error occurred", 0, "None", "None"]

                    else:
                        model_name = "SSIM"
                        print(f"{model_name} is checked")
                        dl_similarity_value = dl_similarity["EfficientNet_FineTuned"]
                        if dl_similarity_value > 0.99:  # Adjust threshold as needed
                            final_decision = f"similar to {valid_img[i]}"
                            flag_decision = 1
                            model_name = "EfficientNet_FineTuned"
                            print(f"{model_name} finds this image similar")
                            return [final_decision, flag_decision, model_name, dl_similarity_value]

                        else:
                            model_name = "EfficientNet_FineTuned_B0"
                            print(f"{model_name} is checked")
                            dl_models_selected = {"MobileNet": mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)}
                            dl_similarity = compute_similarity(img1_path, img2_path, dl_models_selected, transform)
                            dl_similarity_value = dl_similarity["MobileNet"]
                            if dl_similarity_value >0.76:
                                final_decision = f"similar to {valid_img[i]}"
                                flag_decision =1
                                model_name = "MobileNet"
                                print(f"{model_name} finds this image similar")
                                return [final_decision,flag_decision, model_name,dl_similarity_value]
                            else:
                                model_name = "MobileNet"
                                print(f"{model_name} is checked")
                                dl_models_selected = {"VGG16": vgg16(weights=VGG16_Weights.DEFAULT)}
                                dl_similarity = compute_similarity(img1_path, img2_path, dl_models_selected, transform)
                                dl_similarity_value = dl_similarity["VGG16"]
                                if dl_similarity_value >0.9:
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