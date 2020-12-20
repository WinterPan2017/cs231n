'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 
             
             Finished two_layer_net.ipynb here.
             
Author: Winter
Email: 837950571@qq.com
Date: 2020-12-20 15:48:09
LastEditTime: 2020-12-20 15:58:48
'''
import numpy as np
import matplotlib.pyplot as plt
from utils import eval_numerical_gradient
from data_utils import load_CIFAR10


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.parameters = {}
        self.parameters['W1'] = std * np.random.randn(input_size, hidden_size)
        self.parameters['b1'] = np.zeros(hidden_size)
        self.parameters['W2'] = std * np.random.randn(hidden_size, output_size)
        self.parameters['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        W1, b1 = self.parameters['W1'], self.parameters['b1']
        W2, b2 = self.parameters['W2'], self.parameters['b2']
        N, D = X.shape

        def ReLU(X):
            y = X
            y[X < 0] = 0
            return y
        # forward pass
        Z1 = X.dot(W1) + b1  # (N, hidden)
        A1 = ReLU(Z1)
        Z2 = A1.dot(W2) + b2  # (N, C)
        scores = Z2

        if y is None:  # If the targets are not given then jump out, we're done
            return scores

        # compute loss
        scores = scores - np.max(scores, axis=1, keepdims=True)
        scores = np.exp(scores)
        correct_scores = scores[np.arange(N), y]
        scores_sum = np.sum(scores, axis=1)
        loss = np.sum(-1 * np.log(correct_scores / scores_sum)) / N
        loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # back pass, compute gradient
        grads = {}
        dZ2 = scores / scores_sum.reshape(-1, 1)
        dZ2[np.arange(N), y] -= 1  # (N, C)
        dZ2 /= N
        dW2 = A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0)
        dA1 = dZ2.dot(W2.T)
        dZ1 = dA1
        dZ1[A1 <= 0] = 0  # you should use A1 to classify the Relu, not dA1
        dW1 = X.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0)

        dW1 += 2 * reg * W1
        dW2 += 2 * reg * W2

        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['b1'] = db1
        grads['b2'] = db2

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[indices]
            y_batch = y[indices]

            loss, grads = self.loss(X_batch, y_batch, reg=reg)
            loss_history.append(loss)

            for k in grads:
                self.parameters[k] -= learning_rate * grads[k]

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = np.mean(self.predict(X_batch) == y_batch)
                val_acc = np.mean(self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        def ReLU(X):
            y = X
            y[X < 0] = 0
            return y
        Z1 = X.dot(self.parameters['W1']) + self.parameters['b1']
        A1 = ReLU(Z1)
        Z2 = A1.dot(self.parameters['W2']) + self.parameters['b2']

        return np.argmax(Z2, axis=1)


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


if __name__ == "__main__":
    # Create a small net and some toy data to check your implementations.
    # Note that we set the random seed for repeatable experiments.

    input_size = 4
    hidden_size = 10
    num_classes = 3
    num_inputs = 5

    def init_toy_model():
        np.random.seed(0)
        return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

    def init_toy_data():
        np.random.seed(1)
        X = 10 * np.random.randn(num_inputs, input_size)
        y = np.array([0, 1, 2, 2, 1])
        return X, y

    net = init_toy_model()
    X, y = init_toy_data()

    # test forward pass
    scores = net.loss(X)
    print('Your scores:')
    print(scores)
    print()
    print('correct scores:')
    correct_scores = np.asarray([
        [-0.81233741, -1.27654624, -0.70335995],
        [-0.17129677, -1.18803311, -0.47310444],
        [-0.51590475, -1.01354314, -0.8504215],
        [-0.15419291, -0.48629638, -0.52901952],
        [-0.00618733, -0.12435261, -0.15226949]])
    print(correct_scores)
    print()

    print('Difference between your scores and correct scores(The difference should be very small. We get < 1e-7):')
    print(np.sum(np.abs(scores - correct_scores)))

    # test compute loss
    loss, _ = net.loss(X, y, reg=0.05)
    correct_loss = 1.30378789133

    print('Difference between your loss and correct loss(should be very small, we get < 1e-12):')
    print(np.sum(np.abs(loss - correct_loss)))

    # test compute gradient
    loss, grads = net.loss(X, y, reg=0.05)

    for param_name in grads:  # these should all be less than 1e-8 or so
        def f(W): return net.loss(X, y, reg=0.05)[0]
        param_grad_num = eval_numerical_gradient(
            f, net.parameters[param_name], verbose=False)
        print('%s max relative error: %e' %
              (param_name, rel_error(param_grad_num, grads[param_name])))

    # training the net
    net = init_toy_model()
    stats = net.train(X, y, X, y,
                      learning_rate=1e-1, reg=5e-6,
                      num_iters=100, verbose=False)

    print('Final training loss: ', stats['loss_history'][-1])
    plt.plot(stats['loss_history'])
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training Loss history')
    plt.show()

    # use the net on CIFAR10
    Xtr, Ytr, Xte, Yte = load_CIFAR10()
    Xtr = Xtr.reshape(Xtr.shape[0], -1)
    Xte = Xte.reshape(Xte.shape[0], -1)

    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500

    X_train = Xtr[:num_training, ...]
    y_train = Ytr[:num_training, ...]

    X_val = Xtr[num_training:num_training+num_validation, ...]
    y_val = Ytr[num_training:num_training+num_validation, ...]

    X_test = Xte[:num_test]
    y_test = Yte[:num_test]

    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    input_size = 32 * 32 * 3
    hidden_size = 50
    num_classes = 10
    net = TwoLayerNet(input_size, hidden_size, num_classes)
    stats = net.train(X_train, y_train, X_val, y_val, num_iters=1000, batch_size=200,
                      learning_rate=1e-4, learning_rate_decay=0.95, reg=0.25, verbose=True)
    val_acc = np.mean(net.predict(X_val) == y_val)
    print('Validation accuracy: ', val_acc)

    # Plot the loss function and train / validation accuracies
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()
    plt.show()

    from utils import visualize_grid
    W1 = net.parameters['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

    # tune hyperparameters
    best_net = None
    best_val_acc = -1
    results = {}
    learning_rates = [1e-3] # the value is the best I choose
    learning_rate_decays = [0.95]
    regs = [0.23]
    for learning_rate in learning_rates:
        for learning_rate_decay in learning_rate_decays:
            for reg in regs:
                model = TwoLayerNet(input_size, hidden_size, num_classes)
                model.train(X_train, y_train, X_val, y_val, num_iters=5000, learning_rate=learning_rate,
                            learning_rate_decay=learning_rate_decay, reg=reg)
                acc_val = np.mean(model.predict(X_val) == y_val)
                acc_train = np.mean(model.predict(X_train) == y_train)
                results[(learning_rate, learning_rate_decay, reg)] = (
                    acc_train, acc_val)
                if acc_val > best_val_acc:
                    best_net = model
                    best_val_acc = acc_val
    for learning_rate, learning_rate_decay, reg in sorted(results):
        train_accuracy, val_accuracy = results[(learning_rate, learning_rate_decay, reg)]
        print('learning_rate %e learning_rate_decay %e reg %e train accuracy: %f val accuracy: %f' % (
            learning_rate, learning_rate_decay, reg, train_accuracy, val_accuracy))

    test_acc = (best_net.predict(X_test) == y_test).mean()
    print('Test accuracy: ', test_acc)
