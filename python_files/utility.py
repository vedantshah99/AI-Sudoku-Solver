import cv2 as cv
import numpy as np
from keras.models import load_model

#### PREPROCESSING IMAGE
def preProcess(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY
    imgBlur = cv.GaussianBlur(imgGray, (5,5),1)
    imgTreshold = cv.adaptiveThreshold(imgBlur, 255,1,1,11,2) # APPLY ADAPTIVE TRESHOLD
    return imgTreshold

#### FINDING THE BIGGEST CONTOUR
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for c in contours:
        area = cv.contourArea(c)
        if area > 50: #checks that the contour isn't just noise
            perimeter = cv.arcLength(c,True)
            approx = cv.approxPolyDP(c,0.02*perimeter, True) # approxomates number of sides
            if area > max_area and len(approx) ==4:
                biggest = approx
                max_area = area
    return biggest, max_area

####
def reorder(points):
    points = points.reshape((4,2)) # reshaped to 4 rows with 2 columns
    newPoints =  np.zeros((4,1,2), dtype=np.int32) # creates array with 4 rows of 2 columns
    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)] # top left points
    newPoints[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]
    return newPoints

def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

def getPrediction(boxes, model):
    result = []
    for image in boxes:
        # PREPARE IMAGE
        img = np.asarray(image)
        img = img[4:img.shape[0]-4, 4:img.shape[1]-4]
        img = cv.resize(img, (32,32))
        img = img/255
        img = img.reshape(1,32,32,1)
        # GET PREDICTION
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=-1)
        probabilityVlaue = np.amax(predictions)
        # SAVE RESULT
        if probabilityVlaue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

def displayNumbers(img, numbers, color = (0,255,0)):
    wSection = int(img.shape[1]/9)
    hSection = int(img.shape[0]/9)
    for x in range(9):
        for y in range(9):
            if numbers[(y*9)+x] != 0:
                cv.putText(img, str(numbers[(y*9)+x]), (x*wSection+int(wSection/2)-10, int((y+0.8)*hSection)), cv.FONT_HERSHEY_COMPLEX, 1, color, 2, cv.LINE_AA)
    return img

def intializePredictionModel():
    model = load_model('models/digits_model_20.h5')
    return model