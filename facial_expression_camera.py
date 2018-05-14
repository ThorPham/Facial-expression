import numpy as np
import cv2
import keras
from keras.models import load_model
from urllib import request


facial_index = {0:"Angry",1 :"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Surprise",6:"Neutral"}
model = load_model("facial_expression.h5")
def predict(image):
    im = cv2.resize(image,(48,48),interpolation=cv2.INTER_AREA)
    im_reshape = im.reshape((1,48,48,1)).astype(np.float32)/255
    y = model.predict_classes(im_reshape)
    return facial_index[int(y)]


cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
url='http://192.168.0.103:8080/shot.jpg?rnd=756142'

while True:

    # Use urllib to get the image and convert into a cv2 usable format
    imgResp=request.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    frame=cv2.imdecode(imgNp,-1)
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detection = cascade.detectMultiScale(frame_gray,scaleFactor=1.3,minNeighbors=5)
    for (x,y,w,h) in detection:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi = frame_gray[y:y+h,x:x+w]
        if roi.shape[0]<48 or roi.shape[1]<48 :
            pass
        else :
            facial = predict(roi)
            cv2.putText(frame,facial,(int(x),int(y-10)),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),2)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()