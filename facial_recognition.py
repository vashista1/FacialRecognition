
#Importing openCV
#Using openCV and public cascades to recognize frontalface and eye using Viola Jones Algorithm

import cv2


casFace =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
casEye =cv2.CascadeClassifier('haarcascade_eye.xml')
casSmile = cv2.CascadeClassifier('haarcascade_smile.xml')

def grayFrameDetect(gray,frame):
     #Face cascade only uses gray scale image and the 
     #image is shurnk to fasten the search for the matching pixels
     #Along with the pixels in the zone extra pixels are sent to eliminate
     #confusion and better implementation of face recognition
    faces=casFace.detectMultiScale(gray,1.3,5) 
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0),2)
        multiGray = gray[y:y+h, x:x+w]
        multiColor = frame[y:y+h, x:x+w]
        eyes = casEye.detectMultiScale(multiGray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(multiColor, (ex, ey), (ex+ew, ey+eh),(0, 0, 255), 2)
        smiles = casSmile.detectMultiScale(multiGray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(multiColor, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 3)
    return frame

video_capture = cv2.VideoCapture(0)
while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = grayFrameDetect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

video_capture.release()
cv2.destroyAllWindows()

