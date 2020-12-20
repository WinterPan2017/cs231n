'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 

             Finished knn.ipynb here.
             
Author: Winter
Email: 837950571@qq.com
Date: 2020-11-05 15:34:51
LastEditTime: 2020-12-20 15:55:21
'''
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt


class KNearestNeighbor(object):
    def __init__(self, k=1, distance_method='L2'):
        self.k = k
        self.distance_method = distance_method

    def train(self, Xtr, Ytr):
        self.Xtr = Xtr
        self.Ytr = Ytr

    def predict(self, Xte):
        dists = None
        if self.distance_method == 'L1':
            dists = self.__compute_L1_distance(Xte)
        elif self.distance_method == 'L2':
            dists = self.__compute_L2_distance(Xte)
        return self.__predict_labels(dists, self.k)

    def __compute_L1_distance(self, Xte):
        return np.sqrt(
            np.sum(Xte**2, axis=1, keepdims=True)
            + np.sum(self.Xtr**2, axis=1, keepdims=True).T
            - 2 * (Xte @ self.Xtr.T)
        )

    def __compute_L2_distance(self, Xte):
        return (
            np.sum(Xte**2, axis=1, keepdims=True)
            + np.sum(self.Xtr**2, axis=1, keepdims=True).T
            - 2 * (Xte @ self.Xtr.T)
        )

    def __predict_labels(self, dists, k):
        test_num = dists.shape[0]
        y_pred = np.zeros(test_num)
        pred_K = self.Ytr[np.argsort(dists, axis=1)[..., :k]]
        print(pred_K.shape)
        for i in range(test_num):
            y_pred[i] = np.argmax(np.bincount(pred_K[i]))
        return y_pred


def acc(Y_pred, Y_true):
    return np.sum(Y_pred == Y_true)/Y_pred.shape[0]


if __name__ == "__main__":
    Xtr, Ytr, Xte, Yte = load_CIFAR10()
    Xtr = Xtr.reshape(Xtr.shape[0], -1)
    Xte = Xte.reshape(Xte.shape[0], -1)
    # KNN = KNearestNeighbor(k=5)
    # KNN.train(Xtr, Ytr)
    # Ypr = KNN.predict(Xte)
    # print('acc=', acc(Ypr, Yte), '%')

    num_folds = 5
    X_train_folds = np.array_split(Xtr, num_folds)
    Y_train_folds = np.array_split(Ytr, num_folds)

    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    k_choices = [1, 3, 5]
    k_to_accuracies = {}

    for k in k_choices:
        acc = []
        for i in range(num_folds):
            X_train = []
            Y_train = []
            for j in range(num_folds):
                if i != j:
                    X_train.append(X_train_folds[j])
                    Y_train.append(Y_train_folds[j])
            X_train = np.concatenate(X_train, axis=0)
            Y_train = np.concatenate(Y_train, axis=0)
            X_test = X_train_folds[i]
            Y_test = Y_train_folds[i]

            classifier = KNearestNeighbor(k=k)
            classifier.train(X_train, Y_train)
            Y_pre = classifier.predict(X_test)
            acc .append(np.sum(Y_pre == Y_test)/len(Y_test))
        print('k=%d' % (k,))
        print(acc, '\n')
        k_to_accuracies[k] = acc

    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))

    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)
    accuracies_mean = np.array([np.mean(v)
                                for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v)
                               for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()
