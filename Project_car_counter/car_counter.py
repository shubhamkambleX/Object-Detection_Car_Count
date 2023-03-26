import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from sort import *

# cap = cv2.VideoCapture(0)
# cap.set(3,1080)
# cap.set(4,720)


cap = cv2.VideoCapture("../videos/car.mp4")

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
              "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
              "dog","horse","sheep","cow","elepent","bear","zebra","giraffe","backpack","umbrella",
              "handbag","tie","suitcase","frisbee","skis","cell phone","remote"]


mask = cv2.imread("mask_car1.png")

tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limits = [300,297,673,297]

total_counts = []

while True:
    success,img = cap.read()
    results = model(img,stream=True)

    detections = np.empty((0, 5))

    imgRegion = cv2.bitwise_and(img,mask)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            w,h = x2-x1,y2-y1

            conf =math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "car" or currentClass=="bus" or currentClass=="truck"\
                    and conf>.03:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(37, y1)),scale=0.6,thickness=1,offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9)

                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)

    for result in resultsTracker:
        x1,y1,x2,y2,Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(37, y1)),
                           scale=2, thickness=3, offset=10)


        cx,cy = x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)


        if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[1] + 20:
            if total_counts.count(Id) == 0:
                total_counts.append(Id)
            # else:
            #     total_counts += 1

    cvzone.putTextRect(img, f'Count {len(total_counts)}',(50,50))


    cv2.imshow("image",img)
    # cv2.imshow("imageRegion", imgRegion)
    cv2.waitKey(1)
