class CifarImage:
    def __init__(self, data, fine_label, coarse_label, filename):
        """
        Dataset/
        ├── train          # 50,000 training images (batch 1 of 1)
        ├── test           # 10,000 test images
        └── meta          # Contains class names and hierarchical information
        """
        self.data = np.array(data)  # Image data as numpy array
        self.fine_label = fine_label  # Detailed category (0-99)
        self.coarse_label = coarse_label  # Superclass category (0-19)
        self.filename = filename  # Image filename

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict