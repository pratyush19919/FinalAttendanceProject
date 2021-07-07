import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas 






def TrackImages():
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("trained_data\\trainer.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("Student_details.csv")
    print(df)
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    people = set()
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(0 < conf < 75):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == id]['Name'].values
                for i in aa:
                    people.add(i)
                tt=str(id)+"-"+aa
                attendance.loc[len(attendance)] = [id,aa,date,timeStamp]
                
            else:
                id='Unknown'                
                tt=str(id)  
            # if(conf > 75):
            #     noOfFile=len(os.listdir("ImagesUnknown"))+1
            #     cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(conf),(x,y+h), font, 1,(255,255,255),2)    

            cv2.putText(im,str(tt),(x+5,y-5),font,1,(255,255,0),3)
              
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    l_people=list(people)
    
    print(l_people)
    cam.release()
    cv2.destroyAllWindows()

    return l_people


def image(date_arr, att_arr):
    fig,ax = plt.subplots(figsize=(6,6))
    plt.bar(date_arr[0:5], att_arr[0:5] , color ='lightblue',
        width = 0.4)
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom = 0.5)
    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    img64 = base64.b64encode(img.read())
    return img64
    



