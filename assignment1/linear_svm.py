'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 

             Finish svm.ipynb here.
             
Author: Winter
Email: 837950571@qq.com
Date: 2020-11-08 16:25:22
LastEditTime: 2020-12-20 15:57:56
'''
import time
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
from data_utils import load_CIFAR10
from utils import grad_check_sparse
from classifier import LinearClassifier


class LinearSVM(LinearClassifier):
    def loss(self, X_batch, Y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, Y_batch, reg)


def svm_loss_naive(W, X, Y, reg):
    num_train = X.shape[0]
    num_classes = W.shape[1]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[Y[i]]
        for j in range(num_classes):
            if j != Y[i]:
                margin = scores[j] - correct_class_score + 1
                if margin > 0:
                    loss += margin
    loss /= num_train
    loss += reg * np.sum(W*W)  # add regularization term

    dW = np.zeros(W.shape)
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[Y[i]]
        for j in range(num_classes):
            if scores[j] - correct_class_score + 1 > 0 and j != Y[i]:
                dW[..., j] += X[i]
                dW[..., Y[i]] -= X[i]
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, Y, reg):
    loss = 0.0
    num_train = X.shape[0]
    scores = X.dot(W)
    correct_class_score = scores[np.arange(num_train), Y].reshape(-1, 1)
    margin = scores - correct_class_score + 1
    margin[margin <= 0] = 0
    margin[np.arange(num_train), Y] = 0
    loss += np.sum(margin)/num_train
    loss += reg * np.sum(W*W)  # add regularization term

    margin[margin > 0] = 1
    mask = (margin > 0).astype('float')
    mask[np.arange(num_train), Y] = -np.sum(margin, axis=1)
    dW = X.T.dot(mask)
    dW /= num_train
    dW += 2 * reg * W
    return loss, dW


if __name__ == "__main__":
    # load the raw dataset
    Xtr, Ytr, Xte, Yte = load_CIFAR10()
    Xtr = Xtr.reshape(Xtr.shape[0], -1)
    Xte = Xte.reshape(Xte.shape[0], -1)

    # split the dataset into training set, validation set, test set and dev set
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500  # development set, is a subset from traning set

    X_train = Xtr[:num_training, ...]
    Y_train = Ytr[:num_training, ...]

    X_val = Xtr[num_training:num_training+num_validation, ...]
    Y_val = Ytr[num_training:num_training+num_validation, ...]

    X_test = Xte[:num_test]
    Y_test = Yte[:num_test]

    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    Y_dev = Y_train[mask]

    # preprocess each set by subtracting the mean image, then add a bias dimension
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    # test the function for comupting svm loss and gradient
    W = np.random.randn(3073, 10) * 0.0001
    print(svm_loss_naive(W, X_dev, Y_dev, 0.000005))
    print(svm_loss_vectorized(W, X_dev, Y_dev, 0.000005))

    # test the speed on naive way and vectirized way
    tic = time.time()
    loss_naive, grad_naive = svm_loss_naive(W, X_dev, Y_dev, 0.000005)
    toc = time.time()
    print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))

    tic = time.time()
    loss_vectorized, _ = svm_loss_vectorized(W, X_dev, Y_dev, 0.000005)
    toc = time.time()
    print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

    # The losses should match but your vectorized implementation should be much faster.
    print('difference: %f' % (loss_naive - loss_vectorized))

    # # using numerical gradient to check the analytic gradient
    # loss, grad = svm_loss_naive(W, X_dev, Y_dev, 5e1)
    # def f(w): return svm_loss_naive(w, X_dev, Y_dev, 5e1)[0]
    # grad_check_sparse(f, W, grad)

    # test Linear model
    svm = LinearSVM()
    tic = time.time()
    loss_hist = svm.train(X_train, Y_train, learning_rate=1e-7, reg=2.5e4,
                          num_iters=1500, verbose=True)
    toc = time.time()
    print('That took %fs' % (toc - tic))

    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

    Y_train_pred = svm.predict(X_train)
    print('training accuracy: %f' % (np.mean(Y_train == Y_train_pred), ))
    Y_val_pred = svm.predict(X_val)
    print('validation accuracy: %f' % (np.mean(Y_val == Y_val_pred), ))

    # Use validation set to tune hyperparameters
    learning_rates = [1e-7, 5e-5]
    regularization_strengths = [2.5e4, 5e4]

    results = {}
    best_val = -1
    best_svm = None

    for learning_rate in learning_rates:
        for regularization_strength in regularization_strengths:
            model = LinearSVM()
            loss_history = model.train(X_train, Y_train, learning_rate=learning_rate,
                                       reg=regularization_strength, num_iters=1500)
            Y_train_pred = model.predict(X_train)
            acc_train = np.mean(Y_train == Y_train_pred)
            Y_val_pred = model.predict(X_val)
            acc_val = np.mean(Y_val == Y_val_pred)
            results[(learning_rate, regularization_strength)] = (
                acc_train, acc_val)
            if acc_val > best_val:
                best_val = acc_val
                best_svm = model

    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
            lr, reg, train_accuracy, val_accuracy))

    print('best validation accuracy achieved during cross-validation: %f' %
          best_val)  # 0.375000

    # Visualize the cross-validation results
    import math
    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    # plot training accuracy
    marker_size = 100
    colors = [results[x][0] for x in results]
    plt.subplot(2, 1, 1)
    plt.scatter(x_scatter, y_scatter, marker_size,
                c=colors, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 training accuracy')

    # plot validation accuracy
    colors = [results[x][1] for x in results]  # default size of markers is 20
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, marker_size,
                c=colors, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 validation accuracy')
    plt.show()

    # Evaluate the best svm on test set
    y_test_pred = best_svm.predict(X_test)
    test_accuracy = np.mean(Y_test == y_test_pred)
    print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)

    # Visualize the learned weights for each class.
    # Depending on your choice of learning rate and regularization strength, these may
    # or may not be nice to look at.
    w = best_svm.W[:-1, :]  # strip out the bias
    w = w.reshape(32, 32, 3, 10)
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)

        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
        plt.show()
