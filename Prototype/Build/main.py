import helper as hp
import face_mesh_connections as fmc
import main_functions as mf
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

leftEye = {i for pair in fmc.FACEMESH_LEFT_EYE for i in pair}
rightEye = {i for pair in fmc.FACEMESH_RIGHT_EYE for i in pair}
leftIris = {i for pair in fmc.FACEMESH_LEFT_IRIS for i in pair}
rightIris = {i for pair in fmc.FACEMESH_RIGHT_IRIS for i in pair}

min_x, min_y = 10**9, 10**9
max_x, max_y = -1, -1

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
            ## pupil extraction: initialize accumulators
            pupil_x_left = 0.0
            pupil_y_left = 0.0
            pupil_x_right = 0.0
            pupil_y_right = 0.0
            iris_count_left = 0
            iris_count_right = 0
            ## eye conners: initialize boundary points list
            r_outer = faceLandmarks[33]
            r_inner = faceLandmarks[133]
            l_outer = faceLandmarks[263]
            l_inner = faceLandmarks[362]

            for idx, landmark in enumerate(faceLandmarks):
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                if idx in leftEye or idx in rightEye:
                    #cv.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    pass
                elif idx in leftIris:
                    #cv.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    pupil_x_left += x
                    pupil_y_left += y
                    iris_count_left += 1
                elif idx in rightIris:
                    #cv.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    pupil_x_right += x
                    pupil_y_right += y
                    iris_count_right += 1

            ## Calculate the average irises position
            if iris_count_left > 0:
                pupil_x_left = (pupil_x_left / iris_count_left)
                pupil_y_left = (pupil_y_left / iris_count_left)
            if iris_count_right > 0:
                pupil_x_right = (pupil_x_right / iris_count_right)
                pupil_y_right = (pupil_y_right / iris_count_right)

            centroid_pupil_x = (pupil_x_left + pupil_x_right)/2
            centroid_pupil_y = (pupil_y_left + pupil_y_right)/2

            #center_border_x = (i_min_x_point + i_max_x_point) // 2
            #center_border_y = (i_min_y_point + i_max_y_point) // 2

            ## normalizing iris position
            #nx = (centroid_pupil_x - i_min_x_point) / (i_max_x_point - i_min_x_point)
            #ny = (centroid_pupil_y - i_min_y_point) / (i_max_y_point - i_min_y_point)
            h, w, _ = frame.shape
            r_inner_x = int(r_inner.x * w)
            r_inner_y = int(r_inner.y * h)
            r_outer_x = int(r_outer.x * w)
            r_outer_y = int(r_outer.y * h)
            l_inner_x = int(l_inner.x * w)
            l_inner_y = int(l_inner.y * h)
            l_outer_x = int(l_outer.x * w)
            l_outer_y = int(l_outer.y * h)
            r_mid_x = (r_outer_x + r_inner_x) // 2
            r_mid_y = (r_outer_y + r_inner_y) // 2
            l_mid_x = (l_outer_x + l_inner_x) // 2
            l_mid_y = (l_outer_y + l_inner_y) // 2

            anchor_x = (r_mid_x + l_mid_x) // 2
            anchor_y = (r_mid_y + l_mid_y) // 2
            flip_x = True

            ## normalizing for both pupils to anchor (horizontal)
            # right normalization
            r_outer = np.array([r_outer_x, r_outer_y])
            r_inner = np.array([r_inner_x, r_inner_y])
            r_pupil = np.array([pupil_x_right, pupil_y_right])
            r_v = r_inner - r_outer
            r_u = r_pupil - r_outer
            r_nx_x = np.dot(r_u, r_v) / np.dot(r_v, r_v)
            if flip_x is True:
                r_nx_x = 1 - r_nx_x
            # left normalization
            l_outer = np.array([l_outer_x, l_outer_y])
            l_inner = np.array([l_inner_x, l_inner_y])
            l_pupil = np.array([pupil_x_left, pupil_y_left])
            l_v = l_inner - l_outer
            l_u = l_pupil - l_outer
            l_nx_x = np.dot(l_u, l_v) / np.dot(l_v, l_v)
            # averaging 
            r_nx_x = np.clip(r_nx_x, 0.0, 1.0)
            l_nx_x = np.clip(l_nx_x, 0.0, 1.0)
            nx_x = (l_nx_x + r_nx_x) / 2
            #print(l_nx_x, r_nx_x)
            print(nx_x) #check

            ## normalizing for both pupils to anchor (vertical)
            
            

            cv.circle(frame, (r_inner_x, r_inner_y), 1, (0, 0, 255), 1)
            cv.circle(frame, (r_outer_x, r_outer_y), 1, (0, 0, 255), 1)
            cv.circle(frame, (l_inner_x, l_inner_y), 1, (0, 0, 255), 1)
            cv.circle(frame, (l_outer_x, l_outer_y), 1, (0, 0, 255), 1)
            cv.circle(frame, (int(pupil_x_left), int(pupil_y_left)), 1, (255, 255, 255), 5)
            cv.circle(frame, (int(pupil_x_right), int(pupil_y_right)), 1, (255, 255, 255), 5)
            cv.circle(frame, (anchor_x, anchor_y), 1, (0, 0, 255), 5)
            #cv.rectangle(frame, (i_min_x_point, i_min_y_point), (i_max_x_point, i_max_y_point), (0, 255, 0), 2)
            #print(nx," ",ny)

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