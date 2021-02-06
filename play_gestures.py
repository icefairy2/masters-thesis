import os
import random
from datetime import datetime

import cv2

GESTURE_DATASET_ROOT_FOLDER = 'gesture_dataset'
GESTURE_DATASET_FOLDER = os.path.join(GESTURE_DATASET_ROOT_FOLDER, 'organized')

random.seed(datetime.now())

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(GESTURE_DATASET_ROOT_FOLDER, 'stitched.avi'), fourcc, 10, (1280, 960))

for root, dirs, files in os.walk(GESTURE_DATASET_FOLDER):
    if len(files) == 0:
        continue

    idx = random.randint(0, len(files) - 1)
    filepath = os.path.join(root, files[idx])
    cap = cv2.VideoCapture(filepath)

    if not cap.isOpened():
        print("Error opening video  file")

    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (1280, 960), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            cv2.putText(frame, root, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
            cv2.imshow('Frame', frame)

            if cv2.waitKey(50) & 0xFF == ord('p'):
                print(filepath)

            out.write(frame)
        else:
            break

    cap.release()

out.release()
cv2.destroyAllWindows()
