from distutils.errors import PreprocessError
from http.server import ThreadingHTTPServer
import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2 as cv
import pickle

##############################
width = 640
height = 480
threshold = 0.65
##############################

cap = cv.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

model = load_model('digits_model_10.h5')

def preProcessing(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img/255 # normalise image(1-255 1> 0-1)
    return img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv.resize(img, (32,32))
    img = preProcessing(img)
    img = img.reshape(1,32,32,1)
    #Predict
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    classIndex = np.argmax(predictions, axis=1)[0]
    #print(classIndex)
    #print(classIndex[0])

    if probVal > threshold:
        cv.putText(imgOriginal, str(classIndex) + " " + str(probVal),(50,50), cv.FONT_HERSHEY_COMPLEX,1,(0,0,255), 1)
        cv.imshow('Cam View', imgOriginal)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break