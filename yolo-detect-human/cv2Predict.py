# Импорт библиотек
import cv2
from ultralytics import YOLO
import supervision as sv

# Захват камеры
cap = cv2.VideoCapture(0)

# Установка размеров окна
cap.set(3, 1920)
cap.ser(4, 1080)

# Загрузка обученной модели нейронной сети
model = YOLO("./peoples-9/runs/detect/train34/weights/best.pt")

# Установка параметров bboxes
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

while True:
    # Проверка на доступность новых кадров и захват одного кадра
    success, frame = cap.read()

    # Передаем кадр в модель нейронной сети
    result = model(frame)[0]

    # Получение предсказания нейронной сетью
    detections = sv.Detections.from_yolov8(result)

    # Рисуем bboxes на кадре
    frame = box_annotator.annotate(scene=frame, detections=detections)
    
    if success:
        # Кол-во кадров в секунду
        cv2.waitKey(1)

        # Вывод изображения
        cv2.imshow("Result", frame)
