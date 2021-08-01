# from keras.models import load_model
import cv2
import numpy as np
import time
import cv2
import math 
import time
import datetime
import mediapipe as mp
import numpy as np
from pyfirmata import Arduino, SERVO, util
import time

# model = load_model('./mask_detection_model_v2/model-017.model')

port = "COM3"
board = Arduino(port)

it = util.Iterator(board)
it.start()

servo9 = board.get_pin('d:9:s')
servo8 = board.get_pin('d:8:s')

servo9.write(90)
servo8.write(90)

labelsDict={0:'MASK',1:'NO MASK'}
colorDict={0:(0,255,0),1:(0,0,255)}

startTimeAll = datetime.datetime.now()
cam = cv2.VideoCapture(1)

mpFaceDetector = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(False, 1, 0.4, 0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

count = 0
limiter = 10
limStart = datetime.datetime.now()

while True:
    start = time.time()
    success, img = cam.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    
    fh, fw, fc = img.shape
    xCenterPoint = (int(fw/2))
    yCenterPoint = (int(fh/2))
    
    faceId = 1
    
    angleOCX = 0
    angleOCY = 0

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            xCoord = []
            yCoord = []
            nx = 0
            ny = 0
            # mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                xCoord.append(x)
                yCoord.append(y)
                if id == 4:
                    nx = x
                    ny = y
            
            xmin = min(xCoord)
            ymin = min(yCoord)
            xmax = max(xCoord)
            ymax = max(yCoord)

            # faceImg = img[ymin:ymax,xmin:xmax]
            # faceImggray=cv2.cvtColor(faceImg,cv2.COLOR_BGR2GRAY)
            # resized=cv2.resize(faceImggray,(100,100))
            # normalized=resized/255.0
            # reshaped=np.reshape(normalized,(1,100,100,1))
            # result=model.predict(reshaped)
            # label=np.argmax(result,axis=1)[0]

            label = 1

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colorDict[label], 2)
            cv2.putText(img, f'{labelsDict[label]}', (xmin, ymax+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, colorDict[label], 1)

            if label == 1:
                w = 14
                px = 142
                d = 80
                f = (px*d)/w
                
                xLengthFace = abs(xmax-xmin)
                yLengthFace = abs(ymax-ymin)

                xLengthFaceCenter = xmin+int(xLengthFace/2)
                distHeadFromCameraCM = int((w*f)/xLengthFace)

                #find distance from center to head
                distHeadFromCenterPX = math.sqrt( (xLengthFaceCenter-xCenterPoint)**2+(ymin-yCenterPoint)**2 )
                distHeadFromCenterCM = int((distHeadFromCenterPX*w)/px)

                #find distance from camera to center
                distCenterFromCameraCM = int(math.sqrt((distHeadFromCameraCM)**2 - (distHeadFromCenterCM)**2))

                #find distance from head to X
                distOtoXinPX = xLengthFaceCenter-xCenterPoint
                distOtoXinCM = int((distOtoXinPX*w)/px)

                #find distance from head to Y
                distOtoYinPX = ymin-yCenterPoint
                distOtoYinCM = int((distOtoYinPX*w)/px)*-1

                #find distance from cam to X
                distCamtoXinCM = int(math.sqrt((distOtoXinCM)**2+(distCenterFromCameraCM)**2))

                #find distance from cam to Y
                distCamtoYinCM = int(math.sqrt((distOtoYinCM)**2+(distCenterFromCameraCM)**2))

                #find angle on OCX (center, cam, X)
                sinOCX = distOtoXinCM/distCamtoXinCM
                angleOCX = int(math.degrees(math.asin(sinOCX)))

                #find angle on OCY (center, cam, X)
                sinOCY = distOtoYinCM/distCamtoYinCM
                angleOCY = int(math.degrees(math.asin(sinOCY)))

                cv2.circle(img, (xCenterPoint,yCenterPoint), radius=2, color=(255, 0, 255), thickness=3)
                cv2.putText(img, f'O', (xCenterPoint+10, yCenterPoint-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

                cv2.circle(img, (xLengthFaceCenter,ymin), radius=2, color=(0, 255, 0), thickness=3)
                cv2.putText(img, f'H', (xLengthFaceCenter+10, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

                # cv2.line(img, (xCenterPoint,yCenterPoint), (xLengthFaceCenter,ymin), (0, 255, 0), thickness=1)

                cv2.putText(img, f'X', (xLengthFaceCenter-3, yCenterPoint - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                cv2.circle(img, (xLengthFaceCenter,yCenterPoint), radius=2, color=(0, 0, 255), thickness=3)

                cv2.putText(img, f'Y', (xCenterPoint+10, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                cv2.circle(img, (xCenterPoint,ymin), radius=2, color=(0, 0, 255), thickness=3)

                #extra info
                cv2.putText(img, f'{distHeadFromCameraCM} Cam-H', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(img, f'{distCenterFromCameraCM} Cam-O', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(img, f'{distHeadFromCenterCM} OH', (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(img, f'{distOtoXinCM} OX', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(img, f'{distCamtoXinCM} Cam-X', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(img, f'{distOtoYinCM} OY', (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(img, f'{distCamtoYinCM} Cam-y', (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(img, f'{angleOCX} deg OCX', (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(img, f'{angleOCY} deg OCY', (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.line(img, (xCenterPoint,0), (xCenterPoint,fh), (255, 255, 0), thickness=1)
                cv2.line(img, (0,yCenterPoint), (fw,yCenterPoint), (255, 255, 0), thickness=1)

    end = time.time()
    totalTime = end -start
    if totalTime > 0 and count%limiter == 0 and count > 0:

        # limEndTime = datetime.datetime.now()
        # limTimeDiff = (limEndTime - limStart)
        # limDur = limTimeDiff.total_seconds()
        
        fps = 1 / totalTime
        

        angleOCY = angleOCY*6
        angleOCX = angleOCX*6

        if abs(angleOCX) > 90:
            angleOCX = 90
        
        if abs(angleOCY) > 90:
            angleOCY = 90

        print(angleOCX, angleOCY)

        pan = angleOCX+90
        tilt = angleOCY+90
        servo9.write(tilt)
        servo8.write(pan)
       
        limStart = time.time()

    count += 1
    cv2.imshow("all", img)
    k = cv2.waitKey(1) & 0xff
    if k==27:
        break

endTimeAll = datetime.datetime.now()
timeDiffAll = (endTimeAll - startTimeAll)
exec = timeDiffAll.total_seconds()
print(exec)
cam.release()


