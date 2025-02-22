def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/train')
data = {k.decode('utf-8'): v for k, v in data.items()}
print(data.keys())
print(data["filenames"][0]) # name of the image file
print(data["batch_label"][0]) # Label of the batch data
print(data["fine_labels"][0]) # Detailed category labels (100 classes)
print(data["coarse_labels"][0]) # Broader category labels (20 classes)
print(data["data"][0]) # The actual image data in flattened format


