import numpy as np
import os
import csv
import sys
from random import shuffle
import sklearn
from sklearn.model_selection import train_test_split
import cv2
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D, Lambda, Activation, Dropout, Flatten, Dense

track1folder = "../driving_data/windows/"
udacityfolder= "../driving_data/udacity_data/"
csvfilename = "driving_log.csv"


dataType = 'mydata'
samples = []

def csvfile_read(filename):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

"""
Function to read training data from udacity or my own data
"""
def read_training_data():

    if dataType == 'udacity':
        csvfile_read(udacityfolder+csvfilename)
        finalfolder = udacityfolder
        print("Using Udacity data")
    elif dataType == 'mydata':
        csvfile_read(track1folder+csvfilename)
        finalfolder = track1folder
        print("Using my training data")
    else:
        sys.exit("No training data selected")

    return finalfolder

#Define generator function
def generator(samples, finalfolder, batch_size=32):
    num_samples = len(samples)

    while 1:
        shuffle(samples)

        for offset in range(0,num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []

            for batch_sample in batch_samples:
                center_name = finalfolder+"IMG/"+ batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                flip_image = np.fliplr(center_image)
                flip_angle = -center_angle
                images.append(flip_image)
                angles.append(flip_angle)

            #Convert to numpy arrays
            X = np.array(images)
            y = np.array(angles)

            yield sklearn.utils.shuffle(X,y)


def train():

    # Read appropriate training data
    finalfolder = read_training_data()

    # Split samples into test and validation
    train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=42)

    # Define generator function
    train_generator = generator(train_samples, finalfolder, batch_size=128)
    validation_generator = generator(validation_samples, finalfolder, batch_size=128)

    # Define model
    model = Sequential()

    #Normalize Image
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

    #Crop Image
    model.add(Cropping2D(cropping=((60,25),(0,0))))

    #Add Convolution layer1 with filter_size=5x5 and feature_map=12
    model.add(Convolution2D(12,5,5,border_mode='valid'))

    #Max Pooling layer 1
    model.add(MaxPooling2D((2,2)))

    #RELU activation layer1
    model.add(Activation('relu'))

    #Add Convolution layer2 with filter_size=5x5 and feature_map=18
    model.add(Convolution2D(18,5,5,border_mode='valid'))

    #RELU activation layer2
    model.add(Activation('relu'))

    #Add Convolution layer3 with filter_size=5x5 and feature_map=24
    model.add(Convolution2D(24,5,5,border_mode='valid'))

    #Max Pooling layer 3
    model.add(MaxPooling2D((2,2)))

    #RELU activation layer3
    model.add(Activation('relu'))

    #Add Convolution layer4 with filter_size=5x5 and feature_map=36
    model.add(Convolution2D(36,5,5,border_mode='valid'))

    #RELU activation layer4
    model.add(Activation('relu'))

    #Add Convolution layer4 with filter_size=3x3 and feature_map=48
    model.add(Convolution2D(48,3,3,border_mode='valid'))

    #RELU activation layer4
    model.add(Activation('relu'))

    #Flatten layer5
    model.add(Flatten())

    #Dense layer6
    model.add(Dense(300))

    #Dense layer7
    model.add(Dense(150))

    #Dense layer8
    model.add(Dense(50))

    #Dense layer9
    model.add(Dense(10))

    #Dense output layer
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*2, validation_data=validation_generator, nb_val_samples = len(validation_samples)*2, nb_epoch=5)

    model.save('model.h5')


# Call train program to start training the model
train()

