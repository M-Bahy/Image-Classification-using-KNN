from cifar_image import CifarImage , unpickle

elephants = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/elephants')
buses = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/buses')

print('Number of elephants:', len(elephants))
print('Number of buses:', len(buses))