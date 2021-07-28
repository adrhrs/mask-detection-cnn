import cv2
import os
import time
import datetime
import mediapipe as mp
import numpy as np

start_time_all = datetime.datetime.now()
cam = cv2.VideoCapture(1)
# cam = cv2.VideoCapture("ryan.mp4")


mpFaceDetector = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(False, 2, 0.4, 0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
while True:
    success, img = cam.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                if id == 9:
                    nx = x
                    ny = y
            
            xmin = min(xCoord)
            ymin = min(yCoord)
            xmax = max(xCoord)
            ymax = max(yCoord)
            # cv2.imshow("face-"+str(faceId), img[ymin:ymax,xmin:xmax])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
            cv2.putText(img, f'{int(faceId)} - {nx},{ny}', (xmin, ymin-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(img, (nx,ny), radius=1, color=(0, 0, 255), thickness=5)
        
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