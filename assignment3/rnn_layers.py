'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 

             Finished RNN_Captioning.ipynb and LSTM_Caption.ipynb here.
             Implement forward and backward pass for RNN and LSTM.

Author: Winter
Email: 837950571@qq.com
Date: 2020-12-15 14:35:47
LastEditTime: 2020-12-20 16:40:51
'''
import numpy as np
"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None

    next_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + b)
    cache = (x, prev_h, Wx, Wh, next_h)

    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None

    x, prev_h, Wx, Wh, next_h = cache
    df = dnext_h * (1 - next_h**2)  # tanh(x)' = 1 - tanh(x)^2
    dx = df.dot(Wx.T)
    dWx = x.T.dot(df)
    dprev_h = df.dot(Wh.T)
    dWh = prev_h.T.dot(df)
    db = df.sum(axis=0)

    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None

    N, T, D = x.shape
    H = b.shape[0]
    h = np.zeros((N, T, H))
    cache = []
    _h = h0
    for t in range(T):
        _h, cache_t = rnn_step_forward(x[:, t, :], _h, Wx, Wh, b)
        h[:, t, :] = _h
        cache.append(cache_t)

    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None

    N, T, H = dh.shape
    D = cache[0][2].shape[0]
    dx = np.zeros((N, T, D))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros(H)
    dh0 = np.zeros((N, H))
    for t in range(T - 1, -1, -1):
        _dx, dh0, _dWx, _dWh, _db = rnn_step_backward(dh[:, t, :] + dh0,
                                                      cache[t])
        dx[:, t, :] = _dx
        dWx += _dWx
        dWh += _dWh
        db += _db

    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    out = W[x.astype(int)]
    cache = (x, W)

    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None

    x, W = cache
    N, T, D = dout.shape
    dW = np.zeros(W.shape)
    np.add.at(dW, x.ravel(), dout.reshape(-1, D))

    return dW


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None

    H = int(b.shape[0] / 4)
    a = x.dot(Wx) + prev_h.dot(Wh) + b
    i = sigmoid(a[:, :H])
    f = sigmoid(a[:, H:2 * H])
    o = sigmoid(a[:, 2 * H:3 * H])
    g = np.tanh(a[:, 3 * H:])

    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)

    cache = (x, prev_h, prev_c, Wx, Wh, a, i, f, o, g, next_c)

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None

    x, prev_h, prev_c, Wx, Wh, a, i, f, o, g, next_c = cache
    H = prev_c.shape[1]

    # tanh(x)' = 1 - tanh(x)^2
    dnext_c += dnext_h * o * (1 - np.tanh(next_c)**2)
    do = dnext_h * np.tanh(next_c)

    dprev_c = dnext_c * f
    df = dnext_c * prev_c
    di = dnext_c * g
    dg = dnext_c * i

    # sigmoid(x)' = sigmoid(x) * (1 - sigmoid(x))
    da = np.zeros_like(a)
    dai = di * sigmoid(a[:, :H]) * (1 - sigmoid(a[:, :H]))
    daf = df * sigmoid(a[:, H:2 * H]) * (1 - sigmoid(a[:, H:2 * H]))
    dao = do * sigmoid(a[:, 2 * H:3 * H]) * (1 - sigmoid(a[:, 2 * H:3 * H]))
    dag = dg * (1 - np.tanh(a[:, 3 * H:])**2)
    da[:, :H] = dai
    da[:, H:2 * H] = daf
    da[:, 2 * H:3 * H] = dao
    da[:, 3 * H:] = dag

    db = da.sum(axis=0)
    dx = da.dot(Wx.T)
    dWx = x.T.dot(da)
    dprev_h = da.dot(Wh.T)
    dWh = prev_h.T.dot(da)

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None

    N, T, D = x.shape
    N, H = h0.shape
    _h = h0
    c = np.zeros_like(h0)
    cache = []
    h = np.zeros((N, T, H))
    for t in range(T):
        _h, c, cache_t = lstm_step_forward(x[:, t, :], _h, c, Wx, Wh, b)
        h[:, t, :] = _h
        cache.append(cache_t)

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None

    N, T, H = dh.shape
    D = cache[0][0].shape[1]
    dh0 = np.zeros((N, H))
    dc = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dWx = np.zeros((D, 4 * H))
    dWh = np.zeros((H, 4 * H))
    db = np.zeros(4 * H)
    for t in range(T - 1, -1, -1):
        dxt, dh0, dc, dWxt, dWht, dbt = lstm_step_backward(
            dh0 + dh[:, t, :, ], dc, cache[t])
        dx[:, t, :] = dxt
        dWx += dWxt
        dWh += dWht
        db += dbt

    return dx, dh0, dWx, dWh, db


if __name__ == '__main__':
    from gradient_check import rel_error, eval_numerical_gradient, eval_numerical_gradient_array
    N, D, H, T = 2, 5, 4, 3
    x = np.linspace(-0.4, 0.6, num=N * T * D).reshape(N, T, D)
    h0 = np.linspace(-0.4, 0.8, num=N * H).reshape(N, H)
    Wx = np.linspace(-0.2, 0.9, num=4 * D * H).reshape(D, 4 * H)
    Wh = np.linspace(-0.3, 0.6, num=4 * H * H).reshape(H, 4 * H)
    b = np.linspace(0.2, 0.7, num=4 * H)

    h, cache = lstm_forward(x, h0, Wx, Wh, b)

    expected_h = np.asarray([[[0.01764008, 0.01823233, 0.01882671, 0.0194232],
                              [0.11287491, 0.12146228, 0.13018446, 0.13902939],
                              [0.31358768, 0.33338627, 0.35304453,
                               0.37250975]],
                             [[0.45767879, 0.4761092, 0.4936887, 0.51041945],
                              [0.6704845, 0.69350089, 0.71486014, 0.7346449],
                              [0.81733511, 0.83677871, 0.85403753,
                               0.86935314]]])

    print('h error: ', rel_error(expected_h, h))

    N, D, T, H = 2, 3, 10, 6

    x = np.random.randn(N, T, D)
    h0 = np.random.randn(N, H)
    Wx = np.random.randn(D, 4 * H)
    Wh = np.random.randn(H, 4 * H)
    b = np.random.randn(4 * H)

    out, cache = lstm_forward(x, h0, Wx, Wh, b)

    dout = np.random.randn(*out.shape)

    dx, dh0, dWx, dWh, db = lstm_backward(dout, cache)

    fx = lambda x: lstm_forward(x, h0, Wx, Wh, b)[0]
    fh0 = lambda h0: lstm_forward(x, h0, Wx, Wh, b)[0]
    fWx = lambda Wx: lstm_forward(x, h0, Wx, Wh, b)[0]
    fWh = lambda Wh: lstm_forward(x, h0, Wx, Wh, b)[0]
    fb = lambda b: lstm_forward(x, h0, Wx, Wh, b)[0]

    dx_num = eval_numerical_gradient_array(fx, x, dout)
    dh0_num = eval_numerical_gradient_array(fh0, h0, dout)
    dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)
    dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)
    db_num = eval_numerical_gradient_array(fb, b, dout)

    print('dx error: ', rel_error(dx_num, dx))
    print('dh0 error: ', rel_error(dh0_num, dh0))
    print('dWx error: ', rel_error(dWx_num, dWx))
    print('dWh error: ', rel_error(dWh_num, dWh))
    print('db error: ', rel_error(db_num, db))