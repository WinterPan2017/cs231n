'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 

             Finished pytorch.ipynb here.
             
Author: Winter
Email: 837950571@qq.com
Date: 2020-12-06 15:49:20
LastEditTime: 2020-12-20 16:26:44
'''
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.datasets as dset
from torch.utils.data import DataLoader, sampler
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

NUM_TRAIN = 49000

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

cifar10_train = dset.CIFAR10('datasets',
                             train=True,
                             download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train,
                          batch_size=64,
                          sampler=sampler.SubsetRandomSampler(
                              range(NUM_TRAIN)))
cifar10_val = dset.CIFAR10('datasets',
                           train=True,
                           download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val,
                        batch_size=64,
                        sampler=sampler.SubsetRandomSampler(
                            range(NUM_TRAIN, 50000)))
cifar10_test = dset.CIFAR10('datasets',
                            train=False,
                            download=True,
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print_every = 100


def two_layer_fc(x, params):
    """
    A fully-connected neural networks; the architecture is:
    NN is fully connected -> ReLU -> fully connected layer.
    Note that this function only defines the forward pass; 
    PyTorch will take care of the backward pass for us.
    
    The input to the network will be a minibatch of data, of shape
    (N, d1, ..., dM) where d1 * ... * dM = D. The hidden layer will have H units,
    and the output layer will produce scores for C classes.
    
    Inputs:
    - x: A PyTorch Tensor of shape (N, d1, ..., dM) giving a minibatch of
      input data.
    - params: A list [w1, w2] of PyTorch Tensors giving weights for the network;
      w1 has shape (D, H) and w2 has shape (H, C).
    
    Returns:
    - scores: A PyTorch Tensor of shape (N, C) giving classification scores for
      the input data x.
    """
    w1, w2 = params
    x = x.view(x.shape[0], -1)

    out = F.relu(x.mm(w1))
    out = out.mm(w2)
    return out


hidden_layer_size = 42
dtype = torch.float32
x = torch.zeros((64, 50), dtype=dtype)
w1 = torch.zeros((50, hidden_layer_size), dtype=dtype)
w2 = torch.zeros(hidden_layer_size, 10, dtype=dtype)
scores = two_layer_fc(x, [w1, w2])
print(scores.size())


def three_layer_convnet(x, params):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?
    
    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None

    x = F.conv2d(x, conv_w1, conv_b1, padding=2)
    x = F.relu(x)
    x = F.conv2d(x, conv_w2, conv_b2, padding=1)
    x = F.relu(x)
    x = x.view(x.size()[0], -1)
    scores = x.mm(fc_w) + fc_b
    
    return scores


x = torch.zeros((64, 3, 32, 32),
                dtype=dtype)  # minibatch size 64, image size [3, 32, 32]

conv_w1 = torch.zeros(
    (6, 3, 5, 5), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
conv_b1 = torch.zeros((6, ))  # out_channel
conv_w2 = torch.zeros(
    (9, 6, 3, 3), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
conv_b2 = torch.zeros((9, ))  # out_channel

# you must calculate the shape of the tensor after two conv layers, before the fully-connected layer
fc_w = torch.zeros((9 * 32 * 32, 10))
fc_b = torch.zeros(10)

scores = three_layer_convnet(x,
                             [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b])
print(scores.size())  # you should see [64, 10]


def random_weight(shape):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    if len(shape) == 2:
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:])
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w


def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)


print(random_weight((3, 5)))


def check_accuracy_part2(loader, model_fn, params):
    """
    Check the accuracy of a classification model.
    
    Inputs:
    - loader: A DataLoader for the data split we want to check
    - model_fn: A function that performs the forward pass of the model,
      with the signature scores = model_fn(x, params)
    - params: List of PyTorch Tensors giving parameters of the model
    
    Returns: Nothing, but prints the accuracy of the model
    """
    split = 'val' if loader.dataset.train else 'test'
    print('Checking accuracy on the %s set' % split)
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.int64)
            scores = model_fn(x, params)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' %
              (num_correct, num_samples, 100 * acc))


def train_part2(model_fn, params, learning_rate):
    """
    Train a model on CIFAR-10.
    
    Inputs:
    - model_fn: A Python function that performs the forward pass of the model.
      It should have the signature scores = model_fn(x, params) where x is a
      PyTorch Tensor of image data, params is a list of PyTorch Tensors giving
      model weights, and scores is a PyTorch Tensor of shape (N, C) giving
      scores for the elements in x.
    - params: List of PyTorch Tensors giving weights for the model
    - learning_rate: Python scalar giving the learning rate to use for SGD
    
    Returns: Nothing
    """
    for t, (x, y) in enumerate(loader_train):
        # Move the data to the proper device (GPU or CPU)
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)

        # Forward pass: compute scores and loss
        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)

        # Backward pass: PyTorch figures out which Tensors in the computational
        # graph has requires_grad=True and uses backpropagation to compute the
        # gradient of the loss with respect to these Tensors, and stores the
        # gradients in the .grad attribute of each Tensor.
        loss.backward()

        # Update parameters. We don't want to backpropagate through the
        # parameter updates, so we scope the updates under a torch.no_grad()
        # context manager to prevent a computational graph from being built.
        with torch.no_grad():
            for w in params:
                w -= learning_rate * w.grad

                # Manually zero the gradients after running the backward pass
                w.grad.zero_()

        if t % print_every == 0:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))
            check_accuracy_part2(loader_val, model_fn, params)
            print()


# hidden_layer_size = 4000
# learning_rate = 1e-2

# w1 = random_weight((3 * 32 * 32, hidden_layer_size))
# w2 = random_weight((hidden_layer_size, 10))

# train_part2(two_layer_fc, [w1, w2], learning_rate)

# learning_rate = 3e-3

# channel_1 = 32
# channel_2 = 16

# conv_w1 = random_weight((channel_1, 3, 5, 5))
# conv_b1 = zero_weight(channel_1)
# conv_w2 = random_weight((channel_2, channel_1, 3, 3))
# conv_b2 = zero_weight(channel_2)
# fc_w = random_weight((channel_2 * 32 * 32, 10))
# fc_b = zero_weight(10)

# params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
# train_part2(three_layer_convnet, params, learning_rate)
# check_accuracy_part2(loader_val, three_layer_convnet, params)
# check_accuracy_part2(loader_test, three_layer_convnet, params)


class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # assign layer objects to class attributes
        self.fc1 = nn.Linear(input_size, hidden_size)
        # nn.init package contains convenient initialization methods
        # http://pytorch.org/docs/master/nn.html#torch-nn-init
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        # forward always defines connectivity
        x = x.view(x.size()[0], -1)
        scores = self.fc2(F.relu(self.fc1(x)))
        return scores


class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1)
        self.fc = nn.Linear(channel_2 * 32 * 32, num_classes)
        torch.nn.init.kaiming_normal_(self.conv1.weight,
                                      a=0,
                                      mode='fan_in',
                                      nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight,
                                      a=0,
                                      mode='fan_in',
                                      nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc.weight,
                                      a=0,
                                      mode='fan_in',
                                      nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.size()[0], -1)
        scores = self.fc(x)

        return scores


x = torch.zeros((64, 3, 32, 32),
                dtype=dtype)  # minibatch size 64, image size [3, 32, 32]
model = ThreeLayerConvNet(in_channel=3,
                          channel_1=12,
                          channel_2=8,
                          num_classes=10)
scores = model(x)
print(scores.size())  # you should see [64, 10]


def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' %
              (num_correct, num_samples, 100 * acc))


def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()


# hidden_layer_size = 4000
# learning_rate = 1e-2
# model = TwoLayerFC(3 * 32 * 32, hidden_layer_size, 10)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# train_part34(model, optimizer)


# learning_rate = 3e-3
# channel_1 = 32
# channel_2 = 16

# model = ThreeLayerConvNet(3, channel_1, channel_2, 10)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# train_part34(model, optimizer)

channel_1 = 32
channel_2 = 16
learning_rate = 1e-2

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

model = nn.Sequential(
    nn.Conv2d(3, channel_1, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(channel_2 * 32 * 32, 10)
)
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                     momentum=0.9, nesterov=True)

train_part34(model, optimizer)