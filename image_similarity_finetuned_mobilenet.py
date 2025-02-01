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
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
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



def logo_similarity_calculation (img1_path, valid_img, valid_img_path, method , model):

    # Call load_models to initialize both PyTorch models
    dl_models = load_models()
    # final_decision = "no similarity"
    # flag_decision = 0 
    img1_path = check_and_convert_svgtopng(img1_path)
    if img1_path is None:
        return {'result':'An error occured'}
    else: 
        all_similarity_result = {}
        all_similarity = {
        'VGG16': [],
        'EfficientNet_B0': [],
        'MobileNet': [],
        'SSIM': []
        }
        # img = preprocess_image(img_path, target_size=(224, 224), remove_metadata=True)
        img = load_img(img1_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0  
        img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
        if img is None:
            # print(f"Skipping comparison due to invalid images: {img1_path}, {img2_path}")
            print(f"Skipping comparison due to invalid images")
            return {"result" : "Invalid image(s)"}
            # img_array = tf.expand_dims(img, axis=0)  # Add batch dimension
            # Predict
        prediction_MobileNet_FT = model.predict(img_array)
        all_similarity_result["MobileNet_FT"] = round(prediction_MobileNet_FT[0][0].tolist(),4)
        
        
        # Prepare a dictionary to accumulate results for each model across all valid images
        
        for i in valid_img:
            
            img2_path = f"{valid_img_path}{i}"
            dl_similarity= compute_similarity(img1_path, img2_path, dl_models, transform)

            # print(i)
            # print(f"all dl similarity are {all_similarity}")
            ssim_score = round(float(compute_ssim(img1_path, img2_path)),4)
            if "error" in dl_similarity:
                print("Skipping invalid comparison.")
                continue
            # Append each score to the result lists
            all_similarity['VGG16'].append(dl_similarity['VGG16'])
            all_similarity['EfficientNet_B0'].append(dl_similarity['EfficientNet_B0'])
            all_similarity['MobileNet'].append(dl_similarity['MobileNet'])
            all_similarity['SSIM'].append(ssim_score)
            # Print intermediate results
            # print(i)
            # print(f"all similarity are {{'VGG16': {dl_similarity['VGG16']}, 'EfficientNet_B0': {dl_similarity['EfficientNet_B0']}, 'MobileNet': {dl_similarity['MobileNet']}, 'SSIM': {ssim_score}}}")

    if method == "Max":
        # all_similarity_result['VGG16'] = all_similarity['VGG16'].max()
        all_similarity_result['EfficientNet_B0'] = max(all_similarity['EfficientNet_B0'])       
        all_similarity_result['MobileNet'] = max(all_similarity['MobileNet'])
        # all_similarity_result['SSIM'] = max(all_similarity['SSIM'])
        all_similarity_result["avg_not_trained"] = round((all_similarity_result['EfficientNet_B0']+ all_similarity_result['MobileNet'])/2 , 4)
    elif method == "Min":
        # all_similarity_result['VGG16'] = min(all_similarity['VGG16'])
        all_similarity_result['EfficientNet_B0'] = min(all_similarity['EfficientNet_B0'])       
        all_similarity_result['MobileNet'] = min(all_similarity['MobileNet'])
        # all_similarity_result['SSIM'] = min(all_similarity['SSIM'])
        all_similarity_result["avg_not_trained"] = round((all_similarity_result['EfficientNet_B0']+ all_similarity_result['MobileNet'])/2 , 4)
    elif method == "Average":
        # all_similarity_result['VGG16'] = mean(all_similarity['VGG16'])
        all_similarity_result['EfficientNet_B0'] = round(sum(all_similarity['EfficientNet_B0'])/len(all_similarity['EfficientNet_B0']) , 4)       
        all_similarity_result['MobileNet'] = round(sum(all_similarity['MobileNet'])/len(all_similarity['MobileNet']) , 4)
        # all_similarity_result['SSIM'] = round(sum(all_similarity['SSIM'])/len(all_similarity['SSIM']) , 4)
        all_similarity_result["avg_not_trained"] = round((all_similarity_result['EfficientNet_B0']+ all_similarity_result['MobileNet'])/2 , 4)
    return all_similarity_result
##########################################################################


def logo_similarity_make_decision(all_similarity_result):

    # all_similarity_result = logo_similarity_calculation (img1_path, valid_img, valid_img_path, method , model)
    similarity_FT = all_similarity_result["MobileNet_FT"]
    avg_similarity = all_similarity_result["avg_not_trained"]


    if similarity_FT >= 0.8 and avg_similarity >= 0.5:
        return "Sure", 1
    elif similarity_FT < 0.8 and similarity_FT >= 0.6 and avg_similarity >= 0.6:
        return "Maybe", 0.75
    elif similarity_FT < 0.6 and similarity_FT >= 0.5 and avg_similarity >= 0.5:
        return "Maybe", 0.5
    elif similarity_FT <= 0.1:
        return "Never ever", 0.0
    else:
        return "Never", 0.25




# def logo_similarity_make_decision (img_path, model):

#     # Preprocess both images (handle all formats, including PNG fixes)

#     try: 
#         img_path = check_and_convert_svgtopng(img_path)
#         if img_path is None:
#             return ["An error is occured" , 0 , "None"]
#         else: 
#             # img = preprocess_image(img_path, target_size=(224, 224), remove_metadata=True)
#             img = load_img(img_path, target_size=(224, 224))
#             img_array = img_to_array(img) / 255.0  
#             img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
#             #check the image is invalid or not
#             if img is None:
#                 # print(f"Skipping comparison due to invalid images: {img1_path}, {img2_path}")
#                 print(f"Skipping comparison due to invalid images")
#                 return "Invalid image(s)", 0 , "None"
#             # img_array = tf.expand_dims(img, axis=0)  # Add batch dimension
#             # Predict
#             prediction = model.predict(img_array)
#             if prediction[0][0] > 0.8:
#                 return "Bank Mellat Logo", 1, f'{prediction[0][0]:.4f}'
#             else:
#                 return "Not Logo", 0 , f'{prediction[0][0]:.4f}'
#     except Exception as e:
#         print (e)
#         return "Invalid image(s)", 0 , "None"
