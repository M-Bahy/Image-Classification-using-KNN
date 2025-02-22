import numpy as np
import pickle
from cifar_image import CifarImage

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_pickle(name, data):
    with open(name, 'wb') as file:
        pickle.dump(data, file)

def decode_data(d):
    decoded = {}
    for k, v in d.items():
        key = k.decode('utf-8') if isinstance(k, bytes) else k
        # Handle different types of values
        if isinstance(v, bytes):
            value = v.decode('utf-8')
        elif isinstance(v, list):
            # Decode bytes in lists, but skip if it's image data
            if key != 'data':
                value = [x.decode('utf-8') if isinstance(x, bytes) else x for x in v]
            else:
                value = v
        else:
            value = v
        decoded[key] = value
    return decoded


data = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/train')
data = decode_data(data)
meta = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/meta')
meta = decode_data(meta) # fine_label_names & coarse_label_names

# Create list of CifarImage objects
images = []
for i in range(len(data['data'])):
    image = CifarImage(
        data=data['data'][i],
        fine_label=data['fine_labels'][i],
        coarse_label=data['coarse_labels'][i],
        filename=data['filenames'][i]
    )
    images.append(image)

elephants = []
buses = []
for image in images:
    fine_label = image.fine_label
    coarse_label = image.coarse_label
    if meta['fine_label_names'][fine_label] == 'elephant' and meta['coarse_label_names'][coarse_label] == 'large_omnivores_and_herbivores':
        elephants.append(image)
    if meta['fine_label_names'][fine_label] == 'bus' and meta['coarse_label_names'][coarse_label] == 'vehicles_1':
        buses.append(image)
    

data = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/test')
data = decode_data(data)
images = []
for i in range(len(data['data'])):
    image = CifarImage(
        data=data['data'][i],
        fine_label=data['fine_labels'][i],
        coarse_label=data['coarse_labels'][i],
        filename=data['filenames'][i]
    )
    images.append(image)

for image in images:
    fine_label = image.fine_label
    coarse_label = image.coarse_label
    if meta['fine_label_names'][fine_label] == 'elephant' and meta['coarse_label_names'][coarse_label] == 'large_omnivores_and_herbivores':
        elephants.append(image)
    if meta['fine_label_names'][fine_label] == 'bus' and meta['coarse_label_names'][coarse_label] == 'vehicles_1':
        buses.append(image)

print('Number of elephants:', len(elephants))
print('Number of buses:', len(buses))

save_pickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/elephants', elephants)
save_pickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/buses', buses)

elephants = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/elephants')
buses = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/buses')

print('Number of elephants:', len(elephants))
print('Number of buses:', len(buses))
