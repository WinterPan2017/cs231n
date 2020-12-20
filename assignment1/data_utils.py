'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 
Author: Winter
Email: 837950571@qq.com
Date: 2020-11-04 20:03:46
LastEditTime: 2020-12-20 15:54:05
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt



def load_CIFAR10(ROOT='./datasets/cifar-10-batches-py/'):
    """
    load CIFAR10 dataset

    param:
        ROOT: str, the root path of cifar10 datasset
    retunr:
        (Xtr, Ytr, Xte, Yte)
        Xtr: ndarray, images for training set
        Ytr: ndarray, labels for Xtr
        Xte: ndarray, images for test set
        Yte: ndarray, labels for Xte
    """
    Xs = []
    Ys = []
    for i in range(1, 6):
        X, Y = load_CIFAR10_batch(ROOT + 'data_batch_%d' % (i, ))
        Xs.append(X)
        Ys.append(Y)
    Xtr = np.concatenate(Xs)
    Ytr = np.concatenate(Ys)
    del X, Y
    Xte, Yte = load_CIFAR10_batch(ROOT + 'test_batch')
    return Xtr, Ytr, Xte, Yte


def load_CIFAR10_batch(file):
    with open(file, 'rb') as f:
        rawData = pickle.load(f, encoding='bytes')
        X = np.array(rawData[b'data'])
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(rawData[b'labels'])
        return X, Y


def visualize_dataset(X, Y, classes=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], samples_per_class=10):
    num_classes = len(classes)
    for y, class_name in enumerate(classes):
        idxs = np.flatnonzero(Y == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * samples_per_class + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(class_name)
    plt.show()


if __name__ == "__main__":
    Xtr, Ytr, Xte, Yte = load_CIFAR10()
    print(Xtr.shape, Ytr.shape, Xte.shape, Yte.shape)
    visualize_dataset(Xtr, Ytr)
