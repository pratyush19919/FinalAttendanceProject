import os
from PIL import Image
import cv2
import numpy as np


faces_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default_copy.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_faces_id(path):
    i=1
    count=0
    dirpath=[]
    image_paths=[]
    
    for folder in (os.listdir(path)):
        for i in range(1,201):
            pic_path = path + folder + "/" + str(i) +".jpg"
            image_paths.append(pic_path)

    ids=[]
    faceSamples=[]
    # print("-----------------------------------------------------")
    # print(image_paths)
    # print("-----------------------------------------------------")
    # print(len(image_paths))
    # print("----------------------------------------------------")
    for images in image_paths:
        img=Image.open(images)
        arr=np.array(img,'uint8')
        id = images.split("/")[-2][-1]
   
        faces=faces_cascade.detectMultiScale(arr)
        if len(faces) == 0:
            print("----------------------------------------------------")
            print("OOPS!!", images)
            print("----------------------------------------------------")
        for (x,y,w,h) in faces:
            faceSamples.append(arr[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples, ids
