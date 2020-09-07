import cv2
import os


path = "D:/Projects/Python/IP-Camera-Face-Detection/images_in_2"
imagepath = path+"/15994835313.jpg"

image = cv2.imread(imagepath)

folder_path = os.path.join(os.getcwd(),'test')
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
print(folder_path)


cv2.imwrite(folder_path, image)
#cropped_image = image[92:187, 136:224]
cv2.imshow('cropped image',image)

