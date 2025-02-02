import os
import csv
import shutil
import tensorflow as tf
# Import your similarity functions
from image_similarity_finetuned_mobilenet import (
    logo_similarity_calculation as calculation,
    logo_similarity_make_decision as make_decision
)

# Load your fine-tuned model
model = tf.keras.models.load_model(
    "/home/mahdi/logo_detection_model/data-augmentation_v5/trained_model/mobilenet_finetuned_augv5_modify_datasetv2_rofl_v20.h5"
)

def process_images_in_folders(base_dir, model):
    """
    Iterates over all subfolders in 'base_dir',
    runs the similarity + decision pipeline on each image,
    and copies the image to an output subfolder based on the decision comment.
    Returns a list of dictionaries with results for possible logging to CSV.
    """

    # Where to save the images according to their decision
    save_dir = "/home/mahdi/Phishing_Project/detected_logos/1500_detected"
    # save_dir = "/home/mahdi/Phishing_Project/detected_logos/4"
    os.makedirs(save_dir, exist_ok=True)

    # We'll store each image's results (folder name, file name, decision, etc.) in this list
    results = []

    # Loop over subfolders
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        # Skip if it's not a directory
        if not os.path.isdir(folder_path):
            print(f"Skipping non-folder: {folder_name}")
            continue

        print(f"Processing folder: {folder_name}")
        # Loop over files in this subfolder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Check for valid image extensions
            if not file_path.lower().endswith((".jpg", ".jpeg", ".png", ".svg")):
                print(f"Skipping non-image file: {file_name}")
                continue

            # print(f"Processing file: {file_name}")

            # 1) First get the dictionary with similarity scores
            similarity_result = calculation(file_path, valid_img, valid_img_path, 'Max', model)

            # 2) Then get (comment, decision) from that dictionary
            comment, decision = make_decision(similarity_result)

            # # If the function signals an error or invalid input,
            # # you can skip or handle differently. For example:
            # if similarity_result is None:
            #     print(f"Invalid image or error in calculation: {file_name}")
            #     continue

            # Append to our results list (customize as needed)
            results.append({
                "folder": folder_name,
                "file_name": file_name,
                "comment": comment,
                "decision": decision
            })

            # Save the image in a subfolder named after the comment
            # e.g.: sure -> /flagged_images_thr0.8_v20/Sure/...
            comment_folder = comment.replace(" ", "_")  # handle spaces if any
            class_save_dir = os.path.join(save_dir, comment_folder)
            os.makedirs(class_save_dir, exist_ok=True)

            # Copy the image to the corresponding folder
            shutil.copy(file_path, os.path.join(class_save_dir, file_name))

    return results


# Example usage

# 1) Configure your base directory of images
base_dir = "/home/mahdi/Datasets/Logo_Data_1500/"  # put your top-level folder here
# base_dir = "/home/mahdi/Datasets/4/"
# 2) Provide the references (valid_img, valid_img_path) used by your calculation function
valid_img = [
    "BM_LOGO-00.png",
    "BM_LOGO-01.png",
    "BM_LOGO-02.png",
    "BM_LOGO-03.png",
    "BM_LOGO-04.png",
    "BM_LOGO-05.png"
]
valid_img_path = "/home/mahdi/Phishing_Project/Valid_images/"

# 3) Run the pipeline
results = process_images_in_folders(base_dir, model)

# # 4) Print results in console
# if results:
#     for r in results:
#         print(f"Folder: {r['folder']}, File: {r['file_name']}, "
#               f"Comment: {r['comment']}, Decision: {r['decision']}")
# else:
#     print("No results were generated. Check if images are valid and present in the folder.")

# 5) Optionally save results to CSV
output_csv_path = "/home/mahdi/Phishing_Project/output_result_csv/decision_results_1500.csv"
if results:
    # Decide which headers you want to store
    fieldnames = ["folder", "file_name", "comment", "decision"]
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV results saved to: {output_csv_path}")
else:
    print("No data to write to CSV (results list is empty).")
