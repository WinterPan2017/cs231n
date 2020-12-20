'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 

             Finished FullyConnectedNets.ipynb, BatchNormalization.ipynb,
             Dropout.ipynb and ConvolutionalNetworks.ipynb here.
             Implement forward and backward pass for layers.
             
Author: Winter
Email: 837950571@qq.com
Date: 2020-11-23 20:07:15
LastEditTime: 2020-12-20 16:14:05
'''
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    N = x.shape[0]
    out = x.reshape(N, -1).dot(w) + b
    cache = (x, w, b)

    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    N = x.shape[0]
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(N, -1).T.dot(dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0, x)
    cache = (x)

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache
    dx = dout
    dx[x <= 0] = 0

    return dx


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    z, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(z)

    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    dz = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(dz, fc_cache)

    return dx, dw, db


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N, C = x.shape
    mat = x - x[np.arange(N), y].reshape(-1, 1) + 1
    mat[np.arange(N), y] = 0
    mat[mat <= 0] = 0
    loss = mat.sum() / N

    dx = np.zeros((N, C))
    dx[mat > 0] = 1
    dx[np.arange(N), y] = -np.sum(dx, axis=1)
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N, C = x.shape
    es = np.exp(x)
    sum_es = np.sum(es, axis=1)
    loss = (-1 * x[np.arange(N), y] + np.log(sum_es)).sum() / N

    dx = es / sum_es.reshape(-1, 1)
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        nor_x = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = nor_x * gamma + beta
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        cache = (x, sample_mean, sample_var, eps, nor_x, gamma, beta)

    elif mode == "test":
        nor_x = (x - running_mean) / np.sqrt(running_var + eps)
        out = nor_x * gamma + beta

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    # 这里的计算方式就是作业中所提到的第二种，因此不再写另一种
    N, D = dout.shape
    x, mean, var, eps, nor_x, gamma, beta = cache
    dgamma = np.sum(dout * nor_x, axis=0)
    dbeta = np.sum(dout, axis=0)
    dnor_x = dout * gamma
    sqrt_var_eps = np.sqrt(var + eps)  # 减少运算量
    dvar = -0.5 * np.sum(dnor_x * (x - mean) / np.power(sqrt_var_eps, 3),
                         axis=0)
    dmean = -1 * np.sum(dnor_x / sqrt_var_eps, axis=0) - \
              2 * dvar * np.sum(x - mean, axis=0) / N
    dx = dnor_x / sqrt_var_eps + dmean / N + \
             dvar * 2 * (x - mean) / N
    # dvar = -0.5 * np.sum(dnor_x * (x - mean) / np.power(var + eps, 1.5),
    #                      axis=0)
    # dmean = -1 * np.sum(dnor_x / np.sqrt(var + eps), axis=0) - \
    #           2 * dvar * np.sum(x - mean, axis=0) / N
    # dx = dnor_x / np.sqrt(var + eps) + dmean / N + \
    #          dvar * 2 * (x - mean) / N

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    mode = ln_param.get("mode", 'train')

    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    nor_x = (x - mean) / np.sqrt(var + eps)
    out = nor_x * gamma + beta
    cache = (x, mean, var, eps, nor_x, gamma, beta)

    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    N, D = dout.shape
    x, mean, var, eps, nor_x, gamma, beta = cache
    dgamma = np.sum(dout * nor_x, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)
    dnor_x = dout * gamma
    sqrt_var_eps = np.sqrt(var + eps)  # 减少运算量
    dvar = -0.5 * np.sum(
        dnor_x * (x - mean) / np.power(sqrt_var_eps, 3), axis=1, keepdims=True)
    dmean = -1 * np.sum(dnor_x / sqrt_var_eps, axis=1, keepdims=True) - \
              2 * dvar * np.sum(x - mean, axis=1,keepdims=True) / D
    print(dvar.shape, dmean.shape)
    dx = dnor_x / sqrt_var_eps + dmean / D + \
             dvar * 2 * (x - mean) / D

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        mask = (np.random.rand(*(x.shape)) < p) / p
        out = x * mask

    elif mode == "test":
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        dx = dout * mask

    elif mode == "test":
        dx = dout

    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None

    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_o = int(1 + (H + 2 * pad - HH) / stride)
    W_o = int(1 + (W + 2 * pad - WW) / stride)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    out = np.zeros((N, F, H_o, W_o))
    for n in range(N):
        for hi in range(0, H, stride):
            for wi in range(0, W, stride):
                out[n, :, int(hi/stride), int(wi/stride)]= \
                    w.reshape(F, -1).dot(x_pad[n, :, hi:hi + HH, wi:wi + WW].ravel()) + b

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    x, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, H_o, W_o = dout.shape

    dx_pad = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    dw = np.zeros(w.shape)
    for n in range(N):
        for hi in range(0, H, stride):
            for wi in range(0, W, stride):
                # dx 可能存在重叠部分，因此用累加
                dx_pad[n, :, hi:hi + HH, wi:wi + WW] += \
                    dout[n, :, int(hi/stride), int(wi/stride)].dot(w.reshape(F, -1)).reshape(C, HH, WW)
                dw += dout[n, :, int(hi/stride), int(wi/stride)].reshape(-1,1).dot(\
                    x_pad[n, :, hi:hi + HH, wi:wi + WW].reshape(1,-1)\
                        ).reshape(w.shape)
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    db = np.sum(dout, axis=(0, 2, 3))

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, W, H = x.shape
    H_o = int(1 + (H - pool_height) / stride)
    W_o = int(1 + (W - pool_width) / stride)
    out = np.zeros((N, C, H_o, W_o))
    for hi in range(0, H, stride):
        for wi in range(0, W, stride):
            out[:, :, int(hi / stride),
                int(wi / stride)] = np.max(x[:, :, hi:hi + pool_height,
                                             wi:wi + pool_width],
                                           axis=(2, 3))

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None

    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, W, H = x.shape

    dx = np.zeros(x.shape)
    for hi in range(0, H, stride):
        for wi in range(0, W, stride):
            dx[:, :, hi:hi + pool_height,
               wi:wi + pool_width] = dout[:, :,
                                          int(hi / stride),
                                          int(wi / stride)].reshape(
                                              N, C, 1, 1)
            m = np.max(x[:, :, hi:hi + pool_height, wi:wi + pool_width],
                       axis=(2, 3),
                       keepdims=True)
            temp = dx[:, :, hi:hi + pool_height, wi:wi + pool_width]
            mask = x[:, :, hi:hi + pool_height, wi:wi + pool_width] < m
            temp[mask] = 0
            dx[:, :, hi:hi + pool_height, wi:wi + pool_width] = temp

    return dx


from fast_layers import *


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    N, C, H, W = x.shape
    x_ = np.transpose(x, (0, 2, 3, 1)).reshape(-1, C)
    out, cache = batchnorm_forward(x_, gamma, beta, bn_param)
    out = np.transpose(out.reshape(N, H, W, C), (0, 3, 1, 2))

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    N, C, H, W = dout.shape
    dout_ = np.transpose(dout, (0, 2, 3, 1)).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_, cache)
    dx = np.transpose(dx.reshape(N, H, W, C), (0, 3, 1, 2))

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, 
    in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)

    N, C, H, W = x.shape
    channels_per_group = int(C / G)
    mean, var = np.zeros((N, C, 1, 1)), np.zeros((N, C, 1, 1))
    for i in range(G):
        idx = channels_per_group * i
        mean[:,idx:idx+channels_per_group] = \
            np.mean(x[:,idx:idx+channels_per_group,:,:], axis=(1, 2,3), keepdims=True)
        var[:,idx:idx+channels_per_group] = \
            np.var(x[:,idx:idx+channels_per_group,:,:], axis=(1, 2,3), keepdims=True)
    nor_x = (x - mean) / np.sqrt(var + eps)
    out = nor_x * gamma + beta

    cache = (G, x, mean, var, eps, nor_x, gamma, beta)

    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None


    G, x, mean, var, eps, nor_x, gamma, beta = cache
    N, C, H, W = x.shape
    channels_per_group = int(C / G)
    D = channels_per_group * H * W
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * nor_x, axis=(0, 2, 3), keepdims=True)
    dnor_x = dout * gamma
    sqrt_var_eps = np.sqrt(var + eps)  # 减少运算量
    dvar = np.zeros(var.shape)
    dmean = np.zeros(mean.shape)
    for i in range(G):
        idx = channels_per_group * i
        dvar[:, idx:idx + channels_per_group, :, :] = -0.5 * \
            np.sum((dnor_x * (x - mean) / np.power(sqrt_var_eps, 3))[:, idx:idx + channels_per_group, :, :],
            axis=(1, 2, 3),
            keepdims=True)
        dmean[:, idx:idx + channels_per_group,:,:] = -1 * \
            np.sum((dnor_x / sqrt_var_eps)[:, idx:idx + channels_per_group,:,:], axis=(1, 2,3), keepdims=True) - \
              2 * dvar[:, idx:idx + channels_per_group,:,:] * \
                  np.sum((x - mean)[:, idx:idx + channels_per_group,:,:], axis=(1,2,3),keepdims=True) / D

    dx = dnor_x / sqrt_var_eps + dmean / D + \
             dvar * 2 * (x - mean) / D

    return dx, dgamma, dbeta


if __name__ == '__main__':
    from gradient_check import eval_numerical_gradient_array, eval_numerical_gradient, rel_error
    import matplotlib.pyplot as plt
    from imageio import imread
    from PIL import Image
    from time import time
    np.random.seed(231)
    # Check the training-time forward pass by checking means and variances
    # of features both before and after spatial batch normalization

    N, C, H, W = 2, 6, 4, 5
    G = 2
    x = 4 * np.random.randn(N, C, H, W) + 10
    x_g = x.reshape((N * G, -1))
    print('Before spatial group normalization:')
    print('  Shape: ', x.shape)
    print('  Means: ', x_g.mean(axis=1))
    print('  Stds: ', x_g.std(axis=1))

    # Means should be close to zero and stds close to one
    gamma, beta = np.ones((1, C, 1, 1)), np.zeros((1, C, 1, 1))
    bn_param = {'mode': 'train'}

    out, _ = spatial_groupnorm_forward(x, gamma, beta, G, bn_param)
    out_g = out.reshape((N * G, -1))
    print('After spatial group normalization:')
    print('  Shape: ', out.shape)
    print('  Means: ', out_g.mean(axis=1))
    print('  Stds: ', out_g.std(axis=1))
    np.random.seed(231)
    N, C, H, W = 2, 6, 4, 5
    G = 2
    x = 5 * np.random.randn(N, C, H, W) + 12
    gamma = np.random.randn(1, C, 1, 1)
    beta = np.random.randn(1, C, 1, 1)
    dout = np.random.randn(N, C, H, W)

    gn_param = {}
    fx = lambda x: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0]
    fg = lambda a: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0]
    fb = lambda b: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0]

    dx_num = eval_numerical_gradient_array(fx, x, dout)
    da_num = eval_numerical_gradient_array(fg, gamma, dout)
    db_num = eval_numerical_gradient_array(fb, beta, dout)

    _, cache = spatial_groupnorm_forward(x, gamma, beta, G, gn_param)
    dx, dgamma, dbeta = spatial_groupnorm_backward(dout, cache)
    #You should expect errors of magnitudes between 1e-12~1e-07
    print('dx error: ', rel_error(dx_num, dx))
    print('dgamma error: ', rel_error(da_num, dgamma))
    print('dbeta error: ', rel_error(db_num, dbeta))