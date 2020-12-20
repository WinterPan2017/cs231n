'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 

             Finished FullyConnectedNets.ipynb and ConvolutionalNetworks.ipynb here.
             Implements models.
             
Author: Winter
Email: 837950571@qq.com
Date: 2020-11-24 15:27:03
LastEditTime: 2020-12-20 16:21:30
'''
import numpy as np
from layers import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """
    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        W1 = weight_scale * np.random.randn(input_dim, hidden_dim)
        b1 = np.zeros(hidden_dim)
        W2 = weight_scale * np.random.randn(hidden_dim, num_classes)
        b2 = np.zeros(num_classes)
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """

        z1, fc1_cache = affine_forward(X, self.params['W1'], self.params['b1'])
        a1, relu_cache = relu_forward(z1)
        scores, fc2_cache = affine_forward(a1, self.params['W2'],
                                           self.params['b2'])

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * \
                (np.sum(self.params['W1']**2) +np.sum(self.params['W2']**2))
        da1, dw2, db2 = affine_backward(dscores, fc2_cache)
        dz1 = relu_backward(da1, relu_cache)
        _, dw1, db1 = affine_backward(dz1, fc1_cache)
        dw1 += self.reg * self.params['W1']
        dw2 += self.reg * self.params['W2']
        grads['W1'] = dw1
        grads['b1'] = db1
        grads['W2'] = dw2
        grads['b2'] = db2

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """
    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # Initialize the parameters of the network, storing all values in
        # the self.params dictionary. Store weights and biases for the first layer
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be
        # initialized from a normal distribution centered at 0 with standard
        # deviation equal to weight_scale. Biases should be initialized to zero.

        # When using batch normalization, store scale and shift parameters for the
        # first layer in gamma1 and beta1; for the second layer use gamma2 and
        # beta2, etc. Scale parameters should be initialized to ones and shift
        # parameters should be initialized to zeros.

        dims = hidden_dims
        dims.insert(0, input_dim)
        for i in range(1, len(dims)):
            self.params['W' + str(i)] = weight_scale * np.random.randn(
                dims[i - 1], dims[i])
            self.params['b' + str(i)] = np.zeros(dims[i])
            if self.normalization == "batchnorm":
                self.params['gamma' + str(i)] = np.ones(dims[i])
                self.params['beta' + str(i)] = np.zeros(dims[i])
        self.params['W' + str(len(dims))] = weight_scale * np.random.randn(
            dims[-1], num_classes)
        self.params['b' + str(len(dims))] = np.zeros(num_classes)

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{
                "mode": "train"
            } for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None

        # Implement the forward pass for the fully-connected net, computing
        # the class scores for X and storing them in the scores variable.
        #
        # When using dropout, you'll need to pass self.dropout_param to each
        # dropout forward pass.
        #
        # When using batch normalization, you'll need to pass self.bn_params[0] to
        # the forward pass for the first batch normalization layer, pass
        # self.bn_params[1] to the forward pass for the second batch normalization
        # layer, etc.

        out = X
        caches = {}
        cache_fc, cache_bn, cache_rl, cache_do = None, None, None, None
        for i in range(1, self.num_layers):
            out, cache_fc = affine_forward(out, self.params['W' + str(i)],
                                           self.params['b' + str(i)])
            if self.normalization == "batchnorm":
                out, cache_bn = batchnorm_forward(
                    out, self.params['gamma' + str(i)],
                    self.params['beta' + str(i)], self.bn_params[i - 1])
            out, cache_rl = relu_forward(out)
            if self.use_dropout:
                out, cache_do = dropout_forward(out, self.dropout_param)
            caches[i] = (cache_fc, cache_bn, cache_rl, cache_do)
            x = out
        scores, caches[self.num_layers] = affine_forward(
            out, self.params['W' + str(self.num_layers)],
            self.params['b' + str(self.num_layers)])

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}

        # Implement the backward pass for the fully-connected net. Store the
        # loss in the loss variable and gradients in the grads dictionary. Compute
        # data loss using softmax, and make sure that grads[k] holds the gradients
        # for self.params[k]. Don't forget to add L2 regularization!
        #
        # When using batch/layer normalization, you don't need to regularize the scale
        # and shift parameters.
        #
        # NOTE: To ensure that your implementation matches ours and you pass the
        # automated tests, make sure that your L2 regularization includes a factor
        # of 0.5 to simplify the expression for the gradient.

        loss, dscores = softmax_loss(scores, y)
        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(i)]**2)
        dx, dw, db = affine_backward(dscores, caches[self.num_layers])
        dw += self.reg * self.params['W' + str(self.num_layers)]
        grads['W' + str(self.num_layers)] = dw
        grads['b' + str(self.num_layers)] = db
        for i in range(self.num_layers - 1, 0, -1):
            cache_fc, cache_bn, cache_rl, cache_do = caches[i]
            if self.use_dropout:
                dx = dropout_backward(dx, cache_do)
            dx = relu_backward(dx, cache_rl)
            if self.normalization == "batchnorm":
                dx, dgamma, dbeta = batchnorm_backward(dx, cache_bn)
                grads['gamma' + str(i)], grads['beta' + str(i)] = dgamma, dbeta
            dx, dw, db = affine_backward(dx, cache_fc)
            dw += self.reg * self.params['W' + str(i)]
            grads['W' + str(i)], grads['b' + str(i)] = dw, db

        return loss, grads


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype


        # Initialize weights and biases for the three-layer convolutional    
        # network. Weights should be initialized from a Gaussian centered at 0.0   
        # with standard deviation equal to weight_scale; biases should be          
        # initialized to zero. All weights and biases should be stored in the      
        #  dictionary self.params. Store weights and biases for the convolutional  
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    

        C, H, W = input_dim
        W1 = weight_scale * np.random.randn(num_filters, C, filter_size,
                                            filter_size)
        b1 = np.zeros(num_filters)

        W2 = weight_scale * np.random.randn(int(num_filters * H / 2 * W / 2),
                                            hidden_dim)
        b2 = np.zeros(hidden_dim)

        W3 = weight_scale * np.random.randn(hidden_dim, num_classes)
        b3 = np.zeros(num_classes)

        self.params['W1'], self.params['b1'] = W1, b1
        self.params['W2'], self.params['b2'] = W2, b2
        self.params['W3'], self.params['b3'] = W3, b3

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        # Implement the forward pass for the three-layer convolutional net,  
        # computing the class scores for X and storing them in the scores variable. 

        N = X.shape[0]
        out, conv_relu_pool_cache = conv_relu_pool_forward(
            X, W1, b1, conv_param, pool_param)
        out, fc_relu_cache = affine_relu_forward(out, W2, b2)
        scores, fc_cache = affine_forward(out, W3, b3)


        if y is None:
            return scores

        loss, grads = 0, {}
        # Implement the backward pass for the three-layer convolutional net, 
        # storing the loss and gradients in the loss and grads variables. Compute  
        # data loss using softmax, and make sure that grads[k] holds the gradients 
        # for self.params[k]. Don't forget to add L2 regularization!               
        loss, dout = softmax_loss(scores, y)
        loss += self.reg * ((W1**2).sum() + (W2**2).sum() + (W3**2).sum())

        dout, dW3, db3 = affine_backward(dout, fc_cache)
        dout, dW2, db2 = affine_relu_backward(dout, fc_relu_cache)
        dx, dW1, db1 = conv_relu_pool_backward(dout, conv_relu_pool_cache)

        grads['W1'], grads['b1'] = dW1, db1
        grads['W2'], grads['b2'] = dW2, db2
        grads['W3'], grads['b3'] = dW3, db3

        return loss, grads


if __name__ == '__main__':
    from gradient_check import eval_numerical_gradient, rel_error
    import matplotlib.pyplot as plt
    from solver import Solver
    from data_utils import get_CIFAR10_data

    data = get_CIFAR10_data()
    model = ThreeLayerConvNet()

    np.random.seed(231)

    num_train = 100
    small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
    }

    model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

    solver = Solver(model, data,
                    num_epochs=1, batch_size=50,
                    update_rule='adam',
                    optim_config={
                    'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=20)
    solver.train()

    # Print final training accuracy
    print(
        "Full data training accuracy:",
        solver.check_accuracy(small_data['X_train'], small_data['y_train'])
    )
    # Print final validation accuracy
    print(
        "Full data validation accuracy:",
        solver.check_accuracy(data['X_val'], data['y_val'])
    )

    from vis_utils import visualize_grid

    grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
    plt.imshow(grid.astype('uint8'))
    plt.axis('off')
    plt.gcf().set_size_inches(5, 5)
    plt.show()