import pandas as pd
import numpy as np


# loads MNIST data
def load_mnist_data():
    # load train files
    df = None
    y_train = []

    for i in range(10):
        tmp = pd.read_csv('data/mnist/train%d.txt' % i, header=None, sep=" ")

        # build labels - one hot vector
        hot_vector = [1 if j == i else 0 for j in range(0, 10)]

        for j in range(tmp.shape[0]):
            y_train.append(hot_vector)
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df, tmp])

    train_data = df.to_numpy()
    y_train = np.array(y_train)

    # load test files
    df = None
    y_test = []

    for i in range(10):
        tmp = pd.read_csv('data/mnist/test%d.txt' % i, header=None, sep=" ")

        # build labels - one hot vector
        hot_vector = [1 if j == i else 0 for j in range(0, 10)]
        for j in range(tmp.shape[0]):
            y_test.append(hot_vector)
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df, tmp])

    test_data = df.to_numpy()
    y_test = np.array(y_test)

    return train_data, test_data, y_train, y_test


# used to unpickle CIFAR-10 batch files
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


# loads CIFAR-10 data
def load_cifar_data():
    # train data
    train_data = []
    y_train = []
    for i in range(1, 6):
        dct = unpickle('data/cifar/data_batch_%d' % i)
        data = dct[b'data']
        labels = dct[b'labels']

        # build labels - one hot vector
        for label in labels:
            hot_vector = [0]*10
            hot_vector[label] = 1
            y_train.append(hot_vector)

        train_data.extend(data)

    train_data = np.array(train_data)
    y_train = np.array(y_train)

    # test data
    y_test = []
    dct = unpickle('data/cifar/test_batch')
    test_data = dct[b'data']
    labels = dct[b'labels']

    # build labels - one hot vector
    for label in labels:
        hot_vector = [0]*10
        hot_vector[label] = 1
        y_test.append(hot_vector)

    y_test = np.array(y_test)

    return train_data, test_data, y_train, y_test
