import os
import requests
import gzip
import shutil
import numpy as np
import struct
import csv

data_dir = os.getcwd()

file_urls = {
    "train-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
}

for filename, url in file_urls.items():
    unzipped_file = filename.replace(".gz", "")
    
    if not os.path.exists(unzipped_file):
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, "wb") as f:
                shutil.copyfileobj(response.raw, f)
        print(f"Extracting {filename}...")
        with gzip.open(filename, 'rb') as f_in:
            with open(unzipped_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted: {unzipped_file}")
    else:
        print(f"Already exists: {unzipped_file}")

def read_idx_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)
    return images

def read_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def save_to_csv(images, labels, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for img, label in zip(images, labels):
            writer.writerow([label] + img.tolist())
    print(f"Saved {filename}")


train_images = read_idx_images("train-images-idx3-ubyte")
train_labels = read_idx_labels("train-labels-idx1-ubyte")


save_to_csv(train_images, train_labels, "mnist_train.csv")

test_images = read_idx_images("t10k-images-idx3-ubyte")
test_labels = read_idx_labels("t10k-labels-idx1-ubyte")

save_to_csv(test_images, test_labels, "mnist_test.csv")