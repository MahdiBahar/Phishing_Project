from image_similarity import logo_similarity_make_decision as make_decision

# Replace with your image paths
# img1= 'bankmellat_pic3.jpeg'
# img1 = 'bankghavamin_pic7.jpg'
# img1= 'BM_LOGO-01.png' 
# img1= 'banktejarat_pic8.png'
# img1= 'cat.jpg'
# img1= 'flower.jpg'
# img1 = 'mellal.png'
# img1 = 'bankmellat_pic3.jpeg'
# img1= 'bankmellat_pic5.png'
# img1 = '2073/2073_11.jpg'
# img1 = '5233/5233_1.jpg'
# img1 = '2611/2611_4.svg'
# img1 = '2611/2611_17.jpg'
# img1 = "Iran_province-V0.5.svg"
# img1= "Bank_Mellat_Logo.svg"
# img1 = "bin/241_10.bin"
# img1 = 'test/675_1.jpg' # no detection
# img1 = 'test/3669_39.jpg' # detect in logo3
# img1 =  'test_2/207_1.jpg' # not detect because Thr of mobilenet is 76 / logo 4 detect
# img1 = 'test_2/225_1.jpg' # not detected eff = 0.6 / M : 0.66
# img1 = 'test_2/246_1.jpg' #not detected logo 5 --> eff = 0.64 / M = 0.57
# img1 = 'test_2/302_29.jpg' #not detected / transparent
img1 = 'test_2/309_21.jpg'


# 'BM_LOGO-04.png'
valid_img = ['BM_LOGO-00.png' , 'BM_LOGO-01.png' ,  'BM_LOGO-02.png' , 'BM_LOGO-03.png' , 'BM_LOGO-04.png','BM_LOGO-05.png']
img1_path = f"/home/mahdi/Phishing_Project/images/{img1}"

valid_img_path = '/home/mahdi/Phishing_Project/Valid_images/'

result , flag , model_name , similarity_value = make_decision(img1_path, valid_img, valid_img_path)
print(f"result: {result} with the flag {flag}. model: {model_name} with score {similarity_value}")