import cv2 as cv
from ultralytics import YOLO

class TrackShipsUsingYOLO:
    def __init__(self, best='best.pt', min_conf=0.35, max_iou=0.8):
        self.best = best
        self.min_conf = min_conf
        self.max_iou = max_iou
        self.model = YOLO(self.best)

    def infer(self, image):
        results = self.model.track(image, persist=True, conf=self.min_conf, iou=self.max_iou)[0].cpu()
        classes = results.names
        bboxes = results.boxes.xyxy.cpu().numpy().tolist()
        cls = results.boxes.cls.cpu().numpy().tolist()
        conf = results.boxes.conf.cpu().numpy().tolist()
        ids = results.boxes.id
        if ids is not None:
            ids = ids.cpu().numpy().tolist()
        else:
            ids = [0 for _ in bboxes]
        results = []
        for i, (xmin, ymin, xmax, ymax), p, c in zip(ids, bboxes, conf, cls):
            results.append([xmin, ymin, xmax, ymax, i, classes[c]])
        return results

if __name__ == '__main__':
    VIDEO = r"C:\Users\aicpl\ShipsDatasets\VideoDataset\videos\video_1.mp4"
    WEIGHTS = 'yolov8l.pt'
    cap = cv.VideoCapture(VIDEO)
    model = TrackShipsUsingYOLO(WEIGHTS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.infer(frame)
        for xmin, ymin, xmax, ymax, i, c in results:
            frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            frame = cv.putText(frame, f'ID: {i}, {c}', (xmax, ymax - 10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv.imshow('Frame', frame)
        cv.waitKey(1)
    cap.release()
