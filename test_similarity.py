from image_similarity import make_decision

# Replace with your image paths
# img1= 'bankmellat_pic3.jpeg'
# img1 = 'bankghavamin_pic7.jpg'
# img1= 'BM_LOGO-01.png' 
# img1= 'banktejarat_pic8.png'
# img1= 'cat.jpg'
# img1= 'flower.jpg'
# img1 = 'mellal.png'
img1 = 'bankmellat_pic6.jpeg'
# img1= 'bankmellat_pic5.png'

valid_img = ['BM_LOGO-00.png' , 'BM_LOGO-01.png' ,  'BM_LOGO-02.png' , 'BM_LOGO-03.png' , 'BM_LOGO-04.png', 'BM_LOGO-05.png']
img1_path = f"/home/mahdi/Phishing_Project/images/{img1}"

valid_img_path = '/home/mahdi/Phishing_Project/Valid_images/'

result , flag , model_name = make_decision(img1_path, valid_img, valid_img_path)
print(f"the result is {result} with the flag {flag}. The model_name is {model_name}")