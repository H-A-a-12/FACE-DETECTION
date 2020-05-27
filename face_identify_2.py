import cv2
import numpy as np
from os import listdir   # dictor fetch module
from os.path import isfile,join
data_path='C:/Users/Lenovo/Pictures/harsh/'
onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]
print(onlyfiles)
training_data,labels = [],[]

for i,file in enumerate(onlyfiles):
    image_path= data_path + onlyfiles[i]
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    training_data.append(np.asarray(image, dtype=np.uint8))
    labels.append(i)
labels=np.asarray(labels,dtype=np.int32)

model=cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(training_data),np.asarray(labels))
print(" model training complete")

face_classifier = cv2.CascadeClassifier("C:/Users/Lenovo/Pictures/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml")

def face_detector(img,size= 0.5):
    gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,123),3)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))
    return img,roi

cap= cv2.VideoCapture(0)
while(1):
    ret,frame=cap.read()
    image,face=face_detector(frame)

    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        id,result= recognizer.predict(model)
        print(id)
        cv2.imshow('face',image)
        #print(result)
        #print(result[1])
        '''if result[1] < 50:
            confidence = int(100*(1-(result[1])/300)) #?
            display_string = str(confidence)+'%Confidence it is user'
            cv2.putText(image,display_string,(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
            b=cv2.putText(image,'harsh',(400,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
            print(b)

        if confidence > 80:
            cv2.putText(image,"unlock",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(150,200,255),2)
            cv2.imshow('face',image)
        else:
            cv2.putText(image,"locked",(400,480),cv2.FONT_HERSHEY_COMPLEX,1,(200,255,255),2)
            cv2.imshow('face',image)
           '''
        
    except:
        cv2.putText(image,'face not found',(300,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
        #cv2.putText(image,"locked",(200,450),cv2.FONT_HERSHEY_COMPLEX,1,(200,255,255),2)
        cv2.imshow('face',image)
        pass
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()

        
