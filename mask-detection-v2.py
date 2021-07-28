from keras.models import load_model
import cv2
import numpy as np
import time
import cv2
import math 
import time
import datetime
import mediapipe as mp
import numpy as np

model = load_model('./mask_detection_model_v2/model-017.model')

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}


start_time_all = datetime.datetime.now()
cam = cv2.VideoCapture(1)


mpFaceDetector = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(False, 1, 0.4, 0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
while True:
    success, img = cam.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fh, fw, fc = img.shape
    xCenterPoint = (int(fw/2))
    yCenterPoint = (int(fh/2))
    results = faceMesh.process(imgRGB)
    faceId = 1
    start = time.time()
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

            faceImg = img[ymin:ymax,xmin:xmax]
            faceImggray=cv2.cvtColor(faceImg,cv2.COLOR_BGR2GRAY)
            resized=cv2.resize(faceImggray,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))
            result=model.predict(reshaped)
            label=np.argmax(result,axis=1)[0]

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_dict[label], 2)
            cv2.putText(img, f'{labels_dict[label]}', (xmin, ymax+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_dict[label], 1)

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
    if totalTime > 0:
        fps = 1 / totalTime
        print("FPS: ", fps)

    cv2.imshow("all", img)
    k = cv2.waitKey(1) & 0xff
    if k==27:
        break

end_time_all = datetime.datetime.now()
time_diff_all = (end_time_all - start_time_all)
execution_time = time_diff_all.total_seconds()
print(execution_time)
cam.release()


