import numpy as np
import pickle
from random import shuffle
from cifar_image import CifarImage , unpickle

def predict(x_train,x_test):
    pass

def  train_test_split(elephants,buses,train_size=0.8):
    # shuffle the data
    shuffle(elephants)
    shuffle(buses) 
    # Calculate split point
    elephants_split = int(len(elephants) * train_size)
    buses_split = int(len(buses) * train_size)
    
    # Split the data
    elephants_train = elephants[:elephants_split]
    elephants_test = elephants[elephants_split:]
    buses_train = buses[:buses_split]
    buses_test = buses[buses_split:]
    
    # Combine elephants and buses
    x_train = elephants_train + buses_train
    x_test = elephants_test + buses_test

    shuffle(x_train)
    shuffle(x_test)
    
    return x_train, x_test

if __name__ == "__main__" :

    elephants = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/elephants')
    buses = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/buses')

    x_train , x_test = train_test_split(elephants , buses )

    print('Number of items in training set:', len(x_train))
    print('Number of items in test set:', len(x_test))

    predictions = predict(x_train,x_test)