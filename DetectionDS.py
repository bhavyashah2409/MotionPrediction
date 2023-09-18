import cv2 as cv
from ultralytics import YOLO
from deep_sort_pytorch.deep_sort import DeepSort

class TrackShipsUsingDS:
    def __init__(self, best='best.pt', min_conf=0.35, max_iou_distance=0.8, max_age=70, n_init=5):
        self.best = best
        self.min_conf = min_conf
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.tracker = DeepSort(model_path=r'deep_sort_pytorch\deep_sort\deep\checkpoint\ckpt.t7', max_dist=0.2, min_confidence=self.min_conf, nms_max_overlap=0.5, max_iou_distance=self.max_iou_distance, max_age=self.max_age, n_init=self.n_init, nn_budget=100, use_cuda=True)
        self.model = YOLO(self.best)

    def infer(self, image):
        preds = self.model.predict(image)[0].cpu()
        classes = preds.names
        preds = preds.boxes
        xywhs = preds.xywh
        confs = preds.conf
        oids = preds.cls
        final = self.tracker.update(xywhs, confs, oids, image)
        results = []
        for xmin, ymin, xmax, ymax, i, c in final:
            results.append([xmin, ymin, xmax, ymax, i, classes[c]])
        return results

if __name__ == '__main__':
    VIDEO = r"C:\Users\aicpl\ShipsDatasets\VideoDataset\videos\video_1.mp4"
    WEIGHTS = 'yolov8l.pt'
    cap = cv.VideoCapture(VIDEO)
    model = TrackShipsUsingDS(WEIGHTS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.infer(frame)
        for xmin, ymin, xmax, ymax, i, c in results:
            frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            frame = cv.putText(frame, f'ID: {i}, {c}', (xmin, ymin + 20), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv.imshow('Frame', frame)
        cv.waitKey(1)
    cap.release()
