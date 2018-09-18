from os import listdir
import cv2


images = listdir('input/raw/')

for file in images:
    img = cv2.imread('input/raw/'+file)
    img = cv2.resize(img,(256,256))
    cv2.imwrite('input/256/'+file,img)