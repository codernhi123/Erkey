import helper as hp
import face_mesh_connections as fmc
import cv2 as cv
import numpy as np
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       running_mode=vision.RunningMode.VIDEO,
                                       num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(options)

print("Landmarker created successfully")
#End defining

LiveCapture = cv.VideoCapture(0)

prevTime = time.time()

frameTimestampMs = 0

while True:
    isTrue, frame = LiveCapture.read()
    if isTrue is not True:
        break

    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frameRGB)

    # Check 1: cv.imshow("check1", frameRGB)
    
    frameTimestampMs = int(time.time() * 1000)

    results = landmarker.detect_for_video(mp_image, frameTimestampMs)

    idx = 0
    if results.face_landmarks:
        for faceLandmarks in results.face_landmarks:
            for idx, landmark in enumerate(faceLandmarks):
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                leftEye = sorted(set([i for pair in fmc.FACEMESH_LEFT_EYE for i in pair]))
                rightEye = sorted(set([i for pair in fmc.FACEMESH_RIGHT_EYE for i in pair]))
                leftIris = sorted(set([i for pair in fmc.FACEMESH_LEFT_IRIS for i in pair]))
                rightIris = sorted(set([i for pair in fmc.FACEMESH_RIGHT_IRIS for i in pair]))

                if idx in leftEye or idx in rightEye:
                    cv.circle(frame, (x, y), 1, (0, 255, 0), -1)
                elif idx in leftIris or idx in rightIris:
                    cv.circle(frame, (x, y), 1, (0, 0, 255), -1)

            print(len(faceLandmarks))

    curTime = time.time()
    fps = 1/(curTime - prevTime)
    prevTime = curTime

    cv.putText(
        frame, 
        f"FPS: {int(fps)}",
        (20, 40),
        cv.FONT_HERSHEY_COMPLEX,
        1,
        (0, 0, 255), 
        2
    )

    cv.imshow("Face Mesh", frame)


    if cv.waitKey(1) & 0xFF==ord('d'):
        break

LiveCapture.release()
cv.destroyAllWindows()