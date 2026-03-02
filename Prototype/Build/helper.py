import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=0.15):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    #live video
    #capture.set(3,width)
    #capture.set(4,height)
    return 0

def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimension = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimension)

def rotate(img, angle, rotPoint = None):
    (height, width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2, height//2)
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(img, rotMat, dimensions)
