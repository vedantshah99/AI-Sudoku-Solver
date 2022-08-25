import os
from turtle import width
import cv2 as cv
import numpy as np

from utility import *
from sudoku_solver import solve

#######################################
pathImage = 'res/sudoku9.jpg'
heightImg = 450
widthImg =450
model = intializePredictionModel()
#######################################

#### PREPARE THE IMAGE
img=cv.imread(pathImage)
img=cv.resize(img, (widthImg, heightImg)) # RESIZE IMAGE AS SQUARE
imgBlank = np.zeros((heightImg, widthImg,3),np.uint8)
imgThreshold = preProcess(img)

#### FIND ALL CONTOURS
imgContours = img.copy() # MAKES COPY FOR DISPLAY REASONS
imgBigContour = img.copy()
contours, heirarchy = cv.findContours(imgThreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # FINDS ALL CONTOURS
cv.drawContours(imgContours, contours, -1,(0,255,0), 2) # DRAWS CONTOURS ON IMAGE
 
#### FINDS BIGGEST CONTOURS
corners, maxArea = biggestContour(contours)
if len(corners) != 0:
    corners = reorder(corners)
    cv.drawContours(imgBigContour, corners, -1, (0,255,0), 10)
    pts1 = np.float32(corners)
    pts2 = np.float32([[0,0], [widthImg,0], [0,heightImg], [widthImg, heightImg]])
    matrix = cv.getPerspectiveTransform(pts1,pts2)
    imgWarped = cv.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarped= cv.cvtColor(imgWarped, cv.COLOR_BGR2GRAY)

#### SPLIT IMAGE AND FIND EACH DIGIT
imgSolvedDigits = imgBlank.copy()
boxes= splitBoxes(imgWarped)
numbers = getPrediction(boxes, model)
print(numbers)
imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers)
numbers = np.asarray(numbers)
posArray = np.where(numbers>0, 0,1)

#### FIND SOLUTION OF BOARD
board = np.array_split(numbers,9)
try:
    solve(board)
except:
    pass
flatList = []
for row in board:
    for item in row:
        flatList.append(item)
print(posArray)
print(flatList)

solved_array = flatList*posArray
imgSolvedDigits = displayNumbers(imgSolvedDigits, solved_array, color=(0,0,255))

#### CREATING FINAL IMAGE
pts2 = np.float32(corners)
pts1 = np.float32([[0,0],[widthImg, 0],[0,heightImg],[widthImg, heightImg]])
matrix = cv.getPerspectiveTransform(pts1, pts2)
imgInvWarpedColored = img.copy()
imgInvWarpedColored = cv.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
inv_perspective = cv.addWeighted(imgInvWarpedColored,1, img, 0.5,1)


cv.imshow('Original', img)
cv.imshow('Threshold' ,imgThreshold)
cv.imshow('Image contours', imgContours)
cv.imshow('Warped', imgWarped)
cv.imshow('Sample Box',  boxes[0])
cv.imshow('Detected Digits', imgDetectedDigits)
cv.imshow('Solved Array', imgSolvedDigits)
cv.imshow('Final', inv_perspective)

cv.imshow('Detected Digits', imgDetectedDigits)
cv.waitKey(0)