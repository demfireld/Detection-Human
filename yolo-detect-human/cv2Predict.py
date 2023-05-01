import cv2
from ultralytics import YOLO
import supervision as sv

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.ser(4, 1080)
model = YOLO("./peoples-9/runs/detect/train34/weights/best.pt")

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

while True:
    success, frame = cap.read()

    result = model(frame)[0]
    detections = sv.Detections.from_yolov8(result)

    frame = box_annotator.annotate(scene=frame, detections=detections)
    if success:
        cv2.waitKey(1)
        cv2.imshow("Result", frame)
