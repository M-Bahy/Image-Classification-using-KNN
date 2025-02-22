import numpy as np
import matplotlib.pyplot as plt
from cifar_image import CifarImage , unpickle

elephants = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/elephants')
buses = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/buses')

print('Number of elephants:', len(elephants))
print('Number of buses:', len(buses))

def draw_cifar_image(image_array):
    """
    Converts a CIFAR-100 image from 1D array format to RGB format and displays it.
    
    Args:
        image_array: 1D numpy array of shape (3072,) containing the image data
                    [R R R ... G G G ... B B B] format (1024 values each)
    """
    # Reshape the array into 3 channels
    image_array = image_array.reshape(3, 32, 32)
    
    # Reorder from (channel, height, width) to (height, width, channel)
    image_array = image_array.transpose(1, 2, 0)
    
    # Display the image
    plt.imshow(image_array)
    plt.axis('off')  # Hide axes
    plt.show()

# Display an elephant
elephant = elephants[500]
print('Fine label:', elephant.fine_label)
print('Coarse label:', elephant.coarse_label)
print('Filename:', elephant.filename)
draw_cifar_image(elephant.data)