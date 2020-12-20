'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 
             
             Finished softmax.ipynb here.
             
Author: Winter
Email: 837950571@qq.com
Date: 2020-11-10 09:22:03
LastEditTime: 2020-12-20 15:57:06
'''
import numpy as np
from data_utils import load_CIFAR10
from utils import grad_check_sparse
from classifier import LinearClassifier
import matplotlib.pyplot as plt

def softmax_loss(W, X, Y, reg):
    num_train = X.shape[0]
    scores = X.dot(W)
    # Numerical Stability
    scores = scores - np.max(scores, axis=1, keepdims=True)
    scores = np.exp(scores)
    scores_sum = np.sum(scores, axis=1)
    correct_scores = scores[np.arange(num_train), Y]
    loss = np.sum(-1 * np.log(correct_scores/scores_sum)) / num_train
    loss += reg*np.sum(W * W)

    dW = np.zeros(W.shape)
    mask = scores / scores_sum.reshape(-1, 1)
    mask[np.arange(num_train), Y] -= 1
    dW = X.T.dot(mask) / num_train
    dW += 2 * reg * W

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    def loss(self, X_batch, Y_batch, reg):
        return softmax_loss(self.W, X_batch, Y_batch, reg)


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

    # Generate a random softmax weight matrix and use it to compute the loss.
    W = np.random.randn(3073, 10) * 0.0001
    loss, grad = softmax_loss(W, X_dev, Y_dev, 0.0)
    print('loss: %f' % loss)
    # As a rough sanity check, our loss should be something close to -log(0.1).
    print('sanity check: %f' % (-np.log(0.1)))

    # gradient check with regularization
    loss, grad = softmax_loss(W, X_dev, Y_dev, 5e1)
    def f(w): return softmax_loss(w, X_dev, Y_dev, 5e1)[0]
    grad_check_sparse(f, W, grad, 10)

    # Use the validation set to tune hyperparameters
    results = {}
    best_val = -1
    best_softmax = None
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]
    for learing_rate in learning_rates:
        for regularization_strength in regularization_strengths:
            model = SoftmaxClassifier()
            model.train(X_train, Y_train, learning_rate=learing_rate,
                        reg=regularization_strength, num_iters=1500)
            Y_train_pred = model.predict(X_train)
            acc_train = np.mean(Y_train == Y_train_pred)
            Y_val_pred = model.predict(X_val)
            acc_val = np.mean(Y_val_pred == Y_val)
            results[(learing_rate, regularization_strength)] = (acc_train, acc_val)
            if acc_val > best_val:
                best_val = acc_val
                best_softmax = model
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                    lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during cross-validation: %f' % best_val)


    # Evaluate the best softmax on test set
    y_test_pred = best_softmax.predict(X_test)
    test_accuracy = np.mean(Y_test == y_test_pred)
    print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))

    # Visualize the learned weights for each class
    w = best_softmax.W[:-1,:] # strip out the bias
    w = w.reshape(32, 32, 3, 10)

    w_min, w_max = np.min(w), np.max(w)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        
        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
        plt.show()