import cv2
from ultralytics import YOLO
# nano version
# model = YOLO('../Yolo-Weights/yolov8n.pt')
# results = model("Session-5_yolo/images/1.png",show=True)
# cv2.waitKey(0)

model = YOLO('../Yolo-Weights/yolov8l.pt')
results = model("Session-5_yolo/images/3.png",show=True)
cv2.waitKey(0)
