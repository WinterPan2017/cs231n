'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 
             
             Finished FullyConnectedNets.ipynb here.
             Implement servel optimization update rules.
              
Author: Winter
Email: 837950571@qq.com
Date: 2020-11-25 10:14:22
LastEditTime: 2020-12-20 16:16:54
'''
import numpy as np
"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    mu = config["momentum"]
    lr = config["learning_rate"]
    v = mu * v - lr * dw
    next_w = w + v

    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None

    cache = config["cache"]
    decay_rate = config["decay_rate"]
    learning_rate = config["learning_rate"]
    epsilon = config["epsilon"]

    cache = decay_rate * cache + (1 - decay_rate) * dw**2
    next_w = w - learning_rate * dw / (np.sqrt(cache) + epsilon)

    config["cache"] = cache

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None

    learning_rate = config["learning_rate"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    epsilon = config["epsilon"]
    m = config["m"]
    v = config["v"]
    t = config["t"]

    t += 1  # modify t _before_ using it in any calculations
    m = beta1 * m + (1 - beta1) * dw
    mt = m / (1 - beta1**t)
    v = beta2 * v + (1 - beta2) * (dw**2)
    vt = v / (1 - beta2**t)
    next_w = w - learning_rate * mt / (np.sqrt(vt) + epsilon)

    config["m"] = m
    config["v"] = v
    config["t"] = t

    return next_w, config


if __name__ == "__main__":
    from gradient_check import rel_error
    from models import FullyConnectedNet
    from solver import Solver
    from data_utils import get_CIFAR10_data
    import matplotlib.pyplot as plt

    # test sgd_momentum
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    v = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)

    config = {'learning_rate': 1e-3, 'velocity': v}
    next_w, _ = sgd_momentum(w, dw, config=config)

    expected_next_w = np.asarray(
        [[0.1406, 0.20738947, 0.27417895, 0.34096842, 0.40775789],
         [0.47454737, 0.54133684, 0.60812632, 0.67491579, 0.74170526],
         [0.80849474, 0.87528421, 0.94207368, 1.00886316, 1.07565263],
         [1.14244211, 1.20923158, 1.27602105, 1.34281053, 1.4096]])
    expected_velocity = np.asarray(
        [[0.5406, 0.55475789, 0.56891579, 0.58307368, 0.59723158],
         [0.61138947, 0.62554737, 0.63970526, 0.65386316, 0.66802105],
         [0.68217895, 0.69633684, 0.71049474, 0.72465263, 0.73881053],
         [0.75296842, 0.76712632, 0.78128421, 0.79544211, 0.8096]])

    # Should see relative errors around e-8 or less
    print('next_w error: ', rel_error(next_w, expected_next_w))
    print('velocity error: ', rel_error(expected_velocity, config['velocity']))

    # compare sgd_momentum with sgd
    data = get_CIFAR10_data()
    num_train = 4000
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    solvers = {}

    for update_rule in ['sgd', 'sgd_momentum']:
        print('running with ', update_rule)
        model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)

        solver = Solver(model,
                        small_data,
                        num_epochs=5,
                        batch_size=100,
                        update_rule=update_rule,
                        optim_config={
                            'learning_rate': 5e-3,
                        },
                        verbose=True)
        solvers[update_rule] = solver
        solver.train()
        print()

    plt.subplot(3, 1, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Validation accuracy')
    plt.xlabel('Epoch')

    for update_rule, solver in solvers.items():
        plt.subplot(3, 1, 1)
        plt.plot(solver.loss_history, 'o', label="loss_%s" % update_rule)

        plt.subplot(3, 1, 2)
        plt.plot(solver.train_acc_history,
                 '-o',
                 label="train_acc_%s" % update_rule)

        plt.subplot(3, 1, 3)
        plt.plot(solver.val_acc_history,
                 '-o',
                 label="val_acc_%s" % update_rule)

    for i in [1, 2, 3]:
        plt.subplot(3, 1, i)
        plt.legend(loc='upper center', ncol=4)
    plt.gcf().set_size_inches(15, 15)
    plt.show()

    # test rmsprop
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    cache = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)

    config = {'learning_rate': 1e-2, 'cache': cache}
    next_w, _ = rmsprop(w, dw, config=config)

    expected_next_w = np.asarray(
        [[-0.39223849, -0.34037513, -0.28849239, -0.23659121, -0.18467247],
         [-0.132737, -0.08078555, -0.02881884, 0.02316247, 0.07515774],
         [0.12716641, 0.17918792, 0.23122175, 0.28326742, 0.33532447],
         [0.38739248, 0.43947102, 0.49155973, 0.54365823, 0.59576619]])
    expected_cache = np.asarray(
        [[0.5976, 0.6126277, 0.6277108, 0.64284931, 0.65804321],
         [0.67329252, 0.68859723, 0.70395734, 0.71937285, 0.73484377],
         [0.75037008, 0.7659518, 0.78158892, 0.79728144, 0.81302936],
         [0.82883269, 0.84469141, 0.86060554, 0.87657507, 0.8926]])

    # You should see relative errors around e-7 or less
    print('next_w error: ', rel_error(expected_next_w, next_w))
    print('cache error: ', rel_error(expected_cache, config['cache']))

    # test adam
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    m = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)
    v = np.linspace(0.7, 0.5, num=N * D).reshape(N, D)

    config = {'learning_rate': 1e-2, 'm': m, 'v': v, 't': 5}
    next_w, _ = adam(w, dw, config=config)

    expected_next_w = np.asarray(
        [[-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
         [-0.1380274, -0.08544591, -0.03286534, 0.01971428, 0.0722929],
         [0.1248705, 0.17744702, 0.23002243, 0.28259667, 0.33516969],
         [0.38774145, 0.44031188, 0.49288093, 0.54544852, 0.59801459]])
    expected_v = np.asarray([[
        0.69966,
        0.68908382,
        0.67851319,
        0.66794809,
        0.65738853,
    ], [
        0.64683452,
        0.63628604,
        0.6257431,
        0.61520571,
        0.60467385,
    ], [
        0.59414753,
        0.58362676,
        0.57311152,
        0.56260183,
        0.55209767,
    ], [
        0.54159906,
        0.53110598,
        0.52061845,
        0.51013645,
        0.49966,
    ]])
    expected_m = np.asarray(
        [[0.48, 0.49947368, 0.51894737, 0.53842105, 0.55789474],
         [0.57736842, 0.59684211, 0.61631579, 0.63578947, 0.65526316],
         [0.67473684, 0.69421053, 0.71368421, 0.73315789, 0.75263158],
         [0.77210526, 0.79157895, 0.81105263, 0.83052632, 0.85]])

    # You should see relative errors around e-7 or less
    print('next_w error: ', rel_error(expected_next_w, next_w))
    print('v error: ', rel_error(expected_v, config['v']))
    print('m error: ', rel_error(expected_m, config['m']))

    # compare adam with rmsprop
    learning_rates = {'rmsprop': 1e-4, 'adam': 1e-3}
    solvers = {}
    for update_rule in ['adam', 'rmsprop']:
        print('running with ', update_rule)
        model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)

        solver = Solver(
            model,
            small_data,
            num_epochs=5,
            batch_size=100,
            update_rule=update_rule,
            optim_config={'learning_rate': learning_rates[update_rule]},
            verbose=True)
        solvers[update_rule] = solver
        solver.train()
        print()

    plt.subplot(3, 1, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Validation accuracy')
    plt.xlabel('Epoch')

    for update_rule, solver in list(solvers.items()):
        plt.subplot(3, 1, 1)
        plt.plot(solver.loss_history, 'o', label=update_rule)

        plt.subplot(3, 1, 2)
        plt.plot(solver.train_acc_history, '-o', label=update_rule)

        plt.subplot(3, 1, 3)
        plt.plot(solver.val_acc_history, '-o', label=update_rule)

    for i in [1, 2, 3]:
        plt.subplot(3, 1, i)
        plt.legend(loc='upper center', ncol=4)
        plt.gcf().set_size_inches(15, 15)
    plt.show()