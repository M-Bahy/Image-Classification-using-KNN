import numpy as np
import pickle
from random import shuffle
from cifar_image import CifarImage , unpickle



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

def extract_feature_vector(image):
    """
    Extracts a 64x3 feature vector from a 32x32 image stored in a 1D array of 3072 uint8 values.
    
    Parameters:
    image: numpy array of shape (3072,) representing a 32x32 RGB image.
    
    Returns:
    numpy array of shape (64, 3) representing the mean RGB values of 4x4 blocks.
    """
    image = image.reshape(3, 32, 32)  # Reshape to (3, 32, 32) format (RGB, Height, Width)
    feature_vector = []
    
    for i in range(0, 32, 4):
        for j in range(0, 32, 4):
            block = image[:, i:i+4, j:j+4]  # Extract 4x4 block for each channel
            mean_rgb = block.mean(axis=(1, 2))  # Compute mean across height and width
            feature_vector.append(mean_rgb)
    
    return np.array(feature_vector)

def euclidean_distance(x, y):
    """
    Computes the Euclidean distance between two feature vectors.
    
    Parameters:
    x, y: List of tuples/lists, where each element represents an RGB mean (R, G, B) of a block.
          Example: [(120, 200, 150), (80, 90, 100)]
    
    Returns:
    float: Euclidean distance between x and y
    """
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum((x - y) ** 2))

def predict(x_train,x_test,K=5):
    
    for animal in x_train:
        animal.data = extract_feature_vector(animal.data)
    
    for animal in x_test:
        animal.data = extract_feature_vector(animal.data)

    total_number_of_predictions = len(x_test)
    correct_predictions = 0

    for test_image in x_test:
        distances = []
        for train_image in x_train:
            distance = euclidean_distance(test_image.data, train_image.data)
            distances.append((distance, train_image.fine_label))
        
        distances.sort(key=lambda x: x[0])
        print('Predicted label:', distances[0][1])
        print('Distance:', distances[0][0])
        print()

    accuracy = (correct_predictions / total_number_of_predictions) * 100
    print('Accuracy: {:.2f}%'.format(accuracy))

if __name__ == "__main__" :

    elephants = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/elephants')
    buses = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/buses')

    x_train , x_test = train_test_split(elephants , buses )
    predict(x_train,x_test)