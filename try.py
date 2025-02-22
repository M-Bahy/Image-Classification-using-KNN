def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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
print("----------------- Data -----------------")
"""
Dataset/
├── train          # 50,000 training images (batch 1 of 1)
├── test           # 10,000 test images
└── meta          # Contains class names and hierarchical information
"""
print(data.keys())
print(data["filenames"][0]) # name of the image file
print(data["batch_label"]) # Label of the batch data
print(data["fine_labels"][0]) # Detailed category labels (100 classes)
print(data["coarse_labels"][0]) # Broader category labels (20 classes)
print(data["data"][0]) # The actual image data in flattened format
print(f"Number of training images: {len(data['data'])}")  # Should print 50000





print("----------------- Metadata -----------------")

data = unpickle('/media/bahy/MEDO BAHY/CMS/Deep Learning/Assignment 1/Image-Classification-using-KNN/Dataset/meta')
data = decode_data(data)
print(data.keys())
print(data["fine_label_names"][0]) # The 100 classes
print(data["coarse_label_names"][0]) # The 20 classes



