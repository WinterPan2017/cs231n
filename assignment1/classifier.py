'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 
Author: Winter
Email: 837950571@qq.com
Date: 2020-11-10 10:43:28
LastEditTime: 2020-11-10 10:57:20
'''
import numpy as np


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, X, Y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(Y) + 1
        if self.W == None:
            self.W = 0.01 * np.random.randn(dim, num_classes)

        loss_history = []
        for i in range(num_iters):
            indices = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[indices]
            Y_batch = Y[indices]

            loss, grad = self.loss(X_batch, Y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * grad

            if verbose and i % 100 == 0:
                print('iteration %d / %d: loss %f' % (i, num_iters, loss))

        return loss_history

    def predict(self, X):
        Y_pred = np.argmax(X.dot(self.W), axis=1)
        return Y_pred

    def loss(self, X_batch, Y_batch, reg):
        pass
        return None, None
