from image_similarity_finetuned_mobilenet import logo_similarity_make_decision as make_decision
from image_similarity_finetuned_mobilenet import logo_similarity_calculation as calculation
import tensorflow as tf
import os
import csv



model = tf.keras.models.load_model("/home/mahdi/logo_detection_model/data-augmentation_v5/trained_model/mobilenet_finetuned_augv5_modify_datasetv2_rofl_v20.h5")


# img1 = 'test/675_1.jpg' # no detection
# # 'BM_LOGO-04.png'
# valid_img = ['BM_LOGO-00.png' , 'BM_LOGO-01.png' ,  'BM_LOGO-02.png' , 'BM_LOGO-03.png' , 'BM_LOGO-04.png','BM_LOGO-05.png']
# # img1_path = f"/home/mahdi/Phishing_Project/images/{img1}"

# img1_path = "/home/mahdi/logo_detection_model/flag_images/flagged_images/218/218_29.jpg"
# img1_path = "/home/mahdi/logo_detection_model/Test4/flagged_images/218/218_29.jpg"


# valid_img_path = '/home/mahdi/Phishing_Project/Valid_images/'

# result  = make_decision(img1_path, valid_img, valid_img_path, 'Max' , model)
# print(f"result: {result} ")


# 2) Configure your paths
folder_path = "/home/mahdi/Datasets/Flag_2/0"
valid_img_path = "/home/mahdi/Phishing_Project/Valid_images/"
output_csv_path = "/home/mahdi/Phishing_Project/output_result_csv/similarity_results_Flag2_0_decision.csv"

# 3) The list of valid images (as before)
valid_img = [
    "BM_LOGO-00.png",
    "BM_LOGO-01.png",
    "BM_LOGO-02.png",
    "BM_LOGO-03.png",
    "BM_LOGO-04.png",
    "BM_LOGO-05.png"
]

# 4) Create/overwrite the CSV and write a header
with open(output_csv_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    # Customize the header to match the structure you expect
    # If make_decision returns e.g., {
    #   'VGG16': [...],
    #   'EfficientNet_B0': [...],
    #   'MobileNet': [...],
    #   'SSIM': [...]
    # }
    # you can name columns accordingly:
    writer.writerow(["Input Image", "Mobile_FT","EfficientNet_B0", "MobileNet", "avg_not_trained", "decision" , "comment"])

    # 5) Loop through each file in the folder
    for filename in os.listdir(folder_path):
        # Filter for file extensions you consider images
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".svg")):
            img1_path = os.path.join(folder_path, filename)

            # Call your existing function
            # Note: you mentioned the function signature: make_decision(img1_path, valid_img, valid_img_path, 'Max', model)
            similarity_result = calculation(img1_path, valid_img, valid_img_path, 'Max', model)
            comment , decision =  make_decision(similarity_result)
            # If everything is valid, your function typically returns something like:
            # { 'VGG16': [...], 'EfficientNet_B0': [...], 'MobileNet': [...], 'SSIM': [...] }
            # so we can write directly to the CSV:
            if isinstance(similarity_result, dict):
                writer.writerow([
                    filename,
                    similarity_result.get("MobileNet_FT", "N/A"),
                    similarity_result.get("EfficientNet_B0", "N/A"),
                    similarity_result.get("MobileNet", "N/A"),
                    similarity_result.get("avg_not_trained", "N/A"),
                    decision,
                    comment
                ])

            else:
                # If there's an error or a different structure, you might handle it differently
                writer.writerow([filename, "Error or invalid result", "", "", ""])

print(f"Done! Similarity results saved to: {output_csv_path}")

