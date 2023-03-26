import cv2
from ultralytics import YOLO
import cvzone
import math

# cap = cv2.VideoCapture(0)
# cap.set(3,1080)
# cap.set(4,720)


cap = cv2.VideoCapture("../videos/car.mp4")

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
              "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
              "dog","horse","sheep","cow","elepent","bear","zebra","giraffe","backpack","umbrella",
              "handbag","tie","suitcase","frisbee","skis","cell phone","remote"]


while True:
    success,img = cap.read()
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # print(x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            w,h = x2-x1,y2-y1

            conf =math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "car" or currentClass=="bus" or currentClass=="truck"\
                    and conf>.03:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(37, y1)),scale=0.6,thickness=1,offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
    cv2.imshow("image",img)
    cv2.waitKey(1)
