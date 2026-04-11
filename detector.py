from ultralytics import YOLO


class HumanDetector:
    def __init__(self):
        # Load YOLOv8 model (nano for CPU efficiency)
        self.model = YOLO("yolov8n.pt")

        # Target classes (COCO dataset)
        self.TARGET_CLASSES = [0, 1, 2, 3, 5, 7]
        # 0 = person
        # 1 = bicycle
        # 2 = car
        # 3 = motorcycle
        # 5 = bus
        # 7 = truck

    def detect(self, frame):

        results = self.model(frame, verbose=False)

        detections = []

        for result in results:

            boxes = result.boxes

            for box in boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Filter only selected classes
                if cls_id in self.TARGET_CLASSES and conf > 0.5:

                    detections.append((
                        x1,
                        y1,
                        x2,
                        y2,
                        conf,
                        cls_id
                    ))

        return detections