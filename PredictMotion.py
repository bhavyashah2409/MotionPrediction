import cv2 as cv
import numpy as np
from DetectionDS import TrackShipsUsingDS
from DetectionYOLO import TrackShipsUsingYOLO
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor

VIDEO = r"C:\Users\aicpl\ShipsDatasets\VideoDataset\videos\video_1.mp4"
WEIGHTS = 'yolov8l.pt'
TRACKING_INTERVAL = 70
LAST_N = 50
PREDICTION_INTERVAL = 10
DEGREE = 1

def predict_motion_together(time, data, degree, last_n, pred_int, track_int): # (N, 3)
    data = data[-1 * last_n::pred_int]
    data = np.array(data, dtype='float32') # (N, 3)
    Y = data[:, 1:] # (N, 2)
    X = data[:, 0] # (N,)
    X = np.expand_dims(X, axis=-1) # (N, 1)
    temp = X
    for i in range(2, degree + 1, 1):
        X = np.concatenate([X, temp ** i], axis=-1)
    model = LinearRegression()
    model.fit(X, Y)
    preds = [[i ** j for j in range(1, degree + 1, 1)] for i in range(int(time), int(time + 1000), 1)]
    preds = np.array(preds, dtype='float32')
    preds = model.predict(preds)
    return preds

id_tracks = {}
cap = cv.VideoCapture(VIDEO)
model = TrackShipsUsingDS(WEIGHTS)
f = 0.0
# model = TrackShipsUsingYOLO(WEIGHTS)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_w = frame.shape[1]
    results = model.infer(frame)
    for xmin, ymin, xmax, ymax, i, c in results:
        frame = cv.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        frame = cv.putText(frame, f'ID: {i}, {c}', (int(xmin + 10), int(ymin + 10)), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        x, y = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        if i not in id_tracks:
            id_tracks[i] = [[f, x, y]]
        else:
            id_tracks[i].append([f, x, y])
            if len(id_tracks[i]) > PREDICTION_INTERVAL:
                data = id_tracks[i]
                preds = predict_motion_together(f, data, DEGREE, LAST_N, PREDICTION_INTERVAL, TRACKING_INTERVAL)
                for x, y in preds.tolist():
                    frame = cv.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
            if len(id_tracks[i]) > 2 * TRACKING_INTERVAL:
                for index, (_, x, y) in enumerate(id_tracks[i]):
                    if index % TRACKING_INTERVAL == 0:
                        frame = cv.circle(frame, (int(x), int(y)), 4, (255, 0, 255), -1)
                        _, x0, y0 = id_tracks[i][index - TRACKING_INTERVAL]
                        frame = cv.line(frame, (int(x0), int(y0)), (int(x), int(y)), (255, 0, 0), 2)
                        x0, y0 = x, y
    cv.imshow('Frame', frame)
    cv.waitKey(1)
    f = f + 1
cap.release()
