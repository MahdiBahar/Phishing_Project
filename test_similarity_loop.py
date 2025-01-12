import os
from image_similarity import logo_similarity_make_decision as make_decision
from shutil import copyfile

def process_images(input_folder, valid_img, valid_img_path, output_folder, output_folder_not_detected):
    """
    Process images in a folder, compare them with valid images, and save results.

    Args:
        input_folder (str): Folder containing input images to be processed.
        valid_img (list): List of valid images to compare against.
        valid_img_path (str): Path to the valid images folder.
        output_folder (str): Folder where matched images will be saved.

    Returns:
        None
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_not_detected, exist_ok=True)

    # Process each file in the input folder
    for file_name in os.listdir(input_folder):
        img1_path = os.path.join(input_folder, file_name)

        # Ensure it's an image
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.svg')):
            try:
                print(f"Processing {file_name}...")

                # Run the similarity function
                result, flag, model_name, similarity_value = make_decision(img1_path, valid_img, valid_img_path)
                print(f"result: {result} with the flag {flag}. model: {model_name} with score {similarity_value}")
                if flag == 1:  # Similarity found
                    print(f"Image {file_name} is similar to a valid image.")
                    # Save the matched image to the output folder
                    output_path = os.path.join(output_folder, file_name)
                    copyfile(img1_path, output_path)
                else:
                    print(f"Image {file_name} is not similar to any valid image.")
                    # Save the matched image to the output folder
                    output_path = os.path.join(output_folder_not_detected, file_name)
                    copyfile(img1_path, output_path)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Define paths
input_folder = "/home/mahdi/Phishing_Project/images/test_2"  # Folder containing input images
valid_img = ['BM_LOGO-00.png', 'BM_LOGO-01.png', 'BM_LOGO-02.png', 'BM_LOGO-03.png', 'BM_LOGO-04.png', 'BM_LOGO-05.png']
valid_img_path = '/home/mahdi/Phishing_Project/Valid_images/'  # Folder of valid images
output_folder = "/home/mahdi/Phishing_Project/images/output_images"  # Save matched images here
output_folder_not_detected = "/home/mahdi/Phishing_Project/images/output_images_not_detected"  # Save dismatched images here
# Process images
process_images(input_folder, valid_img, valid_img_path, output_folder, output_folder_not_detected)
