import cv2
import os
import time
import datetime
import mediapipe as mp
import numpy as np

cam = cv2.VideoCapture(1)

mpFaceDetector = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
start_time_all = datetime.datetime.now()

name = input('\n enter name end press <return> ==>  ')

with mpFaceDetector.FaceDetection(min_detection_confidence=0.5) as face_detection:
    counter = 1
    while cam.isOpened():
        success, image = cam.read()
        start = time.time()
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image2)

        if results.detections:
            for id, detection in enumerate(results.detections):
                # mpDraw.draw_detection(image, detection)
                bBox = detection.location_data.relative_bounding_box        
                h, w, c = image2.shape
                
                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                cv2.imwrite("dataset/" + str(name) + '-' + str(counter) + ".jpg", image[boundBox[1]-30:boundBox[1]+boundBox[3]+20,boundBox[0]-10:boundBox[0]+boundBox[2]+10])
                # cv2.putText(image, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
                
        
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        counter += 1
        print("FPS: ", fps, counter)
        # cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        cv2.imshow('test', image)
        # cv2.waitKey(20)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        time.sleep(1)  
        if counter == 20:
            break

end_time_all = datetime.datetime.now()
time_diff_all = (end_time_all - start_time_all)
execution_time = time_diff_all.total_seconds()
print(execution_time)
cam.release()