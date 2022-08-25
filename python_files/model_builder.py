import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle 
import os

##################################
DATADIR = 'res/myData'
NUMBER_OF_CATEGORIES = len(DATADIR)
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2
IMAGE_DIMNESIONS = (32,32,3)
##################################

images = []
classNo = []
categories = os.listdir(DATADIR)
print(categories)

print("IMPORTING IMAGES")
for category in categories:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img = cv.imread(os.path.join(path,img), cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (IMAGE_DIMNESIONS[0],IMAGE_DIMNESIONS[1]))
        images.append(img)
        classNo.append(category)
    print(category,end=" ")
print(" ")

images = np.array(images) # CONVERTS TO NUMPY ARRAY
classNo = np.array(classNo) # SAME THING

#### SPLITTING DATA INTO TRAINGING, TESTING, & VALIDATION
X_train, X_test, y_train, y_test = train_test_split(images, classNo,test_size=TEST_RATIO)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,test_size=VALIDATION_RATIO)

number_of_samples = []
for category in categories:
    number_of_samples.append(len(np.where(y_train==category)[0]))

plt.figure(figsize=(10,5))
plt.bar(range(0,NUMBER_OF_CATEGORIES), number_of_samples)
plt.title("No of Images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
#plt.show()

#### Preprocess all Images
def preProcessing(img):
    img = cv.equalizeHist(img)
    img = img/255 # normalise image(1-255 1> 0-1)
    return img

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2,shear_range=0.1, rotation_range=10)
dataGen.fit(X_train)

y_train = to_categorical(y_train, NUMBER_OF_CATEGORIES)
y_test = to_categorical(y_test, NUMBER_OF_CATEGORIES)
y_validation = to_categorical(y_validation, NUMBER_OF_CATEGORIES)

def myModel(): # BASED ON LINET MODEL
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNodes = 500

    model = Sequential()
    model.add(Conv2D(noOfFilters, sizeOfFilter1,input_shape=(IMAGE_DIMNESIONS[0],IMAGE_DIMNESIONS[1],1),activation='relu'))
    model.add(Conv2D(noOfFilters, sizeOfFilter1,activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Conv2D(noOfFilters//2, sizeOfFilter2,activation='relu'))
    model.add(Conv2D(noOfFilters//2, sizeOfFilter2,activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUMBER_OF_CATEGORIES, activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = myModel()
#print(model.summary())

batchSizeVal = 50
epochsVal = 20
stepsPerEpochVal = len(X_train)//batchSizeVal

history = model.fit_generator(dataGen.flow(X_train,y_train,batch_size=batchSizeVal), steps_per_epoch=stepsPerEpochVal, epochs= epochsVal,validation_data=(X_validation, y_validation),shuffle=1)

score = model.evaluate(X_test, y_test,verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy: ', score[1])

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

save_path = './digits_model_20.h5'
model.save(save_path)


"""
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

pickle_out = open('model_trained_10.p', 'wb')
pickle.dump(model,pickle_out)
pickle_out.close()
"""