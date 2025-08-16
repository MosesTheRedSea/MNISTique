import time
import numpy as np
import random
import matplotlib.pyplot as plt
from plot_utils import plot_curves

def load_csv(path):
    data = []
    labels = []
    with open(path, 'r') as fp:
        for line in fp:
            line = line.rstrip()
            parts = line.split(',')
            labels.append(int(parts[0]))
            pixels = [int(px)/255 for px in parts[1:]]
            data.append(pixels)
    return data, labels

def load_mnist_trainval():
    data, label = load_csv('./mnist_train.csv')
    split_idx = int(len(data) * 0.8) 
    return data[:split_idx], label[:split_idx], data[split_idx:], label[split_idx:]

def load_mnist_test():
    data, label = load_csv('./mnist_test.csv')
    return data, label

def generate_batched_data(data, label, batch_size=32, shuffle=False, seed=None):
    data = np.array(data)
    label = np.array(label)

    if shuffle:
        idx = np.arange(len(data))
        if seed is not None:
            random.seed(seed)
        random.shuffle(idx)
        data = data[idx]
        label = label[idx]

    batched_data = []
    batched_label = []
    for i in range(0, len(data), batch_size):
        batched_data.append(data[i:i+batch_size])
        batched_label.append(label[i:i+batch_size])

    return batched_data, batched_label

def train(epoch, batched_train_data, batched_train_label, model, optimizer):
    epoch_loss = 0.0
    hits = 0
    count_samples = 0

    for inputs, targets in zip(batched_train_data, batched_train_label):
        loss, accuracy = model.forward(inputs, targets)
        optimizer.update(model)
        epoch_loss += loss
        hits += accuracy * inputs.shape[0]
        count_samples += inputs.shape[0]

    return epoch_loss / len(batched_train_data), hits / count_samples

def evaluate(batched_test_data, batched_test_label, model):
    epoch_loss = 0.0
    hits = 0
    count_samples = 0

    for inputs, targets in zip(batched_test_data, batched_test_label):
        loss, accuracy = model.forward(inputs, targets, mode='valid')
        epoch_loss += loss
        hits += accuracy * inputs.shape[0]
        count_samples += inputs.shape[0]

    return epoch_loss / len(batched_test_data), hits / count_samples
