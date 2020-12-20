'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 

             Finished NetworkVisualization-PyTorch,ipynb here.
             
Author: Winter
Email: 837950571@qq.com
Date: 2020-12-16 19:04:23
LastEditTime: 2020-12-20 16:42:57
'''
import torch
import torchvision
import torchvision.transforms as T
from image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import numpy as np
import random


def preprocess(img, size=224):
    """
    Preprocess image
    """
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None])
    ])
    return transform(img)


def deprocess(img, should_rescale=True):
    """
    Deprocess image in contrast of Preprocess
    """
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage()
    ])
    return transform(img)


def rescale(x):
    """
    rescale the data into 0-1
    """
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def blur_image(X, sigma=1):
    """
    bulr image by gaussian fillter
    """
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X


device = torch.device('cpu')


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None

    X.to(device=device)
    y.to(device=device)
    scores = model(X)
    # loss = F.cross_entropy(scores, y)
    scores = scores.gather(1, y.view(-1, 1)).squeeze()
    loss = scores.sum()  # use correct scores sum as loss
    loss.backward()

    saliency = X.grad.max(dim=1).values

    return saliency


def show_saliency_maps(X, y):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    y_tensor = torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()


def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image, and make it require gradient
    X_fooling = X.clone()
    X_fooling = X_fooling.requires_grad_()

    learning_rate = 1

    # When computing an update step, first normalize the gradient:
    #   dX = learning_rate * g / ||g||_2
    X_fooling.to(device=device)
    iteration = 0
    while True:
        scores = model(X_fooling)
        loss = (scores * F.one_hot(torch.tensor(target_y), 1000)).sum()
        loss.backward()
        with torch.no_grad():
            X_fooling += learning_rate * X_fooling.grad / X_fooling.grad.norm()
            pred_y = model(X_fooling).argmax().item()
            sucess = (pred_y == target_y)
            iteration += 1
            print('iter %d, loss=%f, pred=%d, target=%d' %
                  (iteration, loss.item(), pred_y, target_y))
            if sucess:
                break
        model.zero_grad()
        X_fooling.grad.zero_()


    return X_fooling


def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.
    
    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes
    
    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X


def create_class_visualization(target_y, model, dtype, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.
    
    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    - dtype: Torch datatype to use for computations
    
    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to gjitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    model.type(dtype)
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 100)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)

    # Randomly initialize the image as a PyTorch Tensor, and make it requires gradient.
    img = torch.randn(1, 3, 224, 224).mul_(1.0).type(dtype).requires_grad_()

    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
        img.data.copy_(jitter(img.data, ox, oy))

        scores = model(img)
        loss = (scores * F.one_hot(torch.tensor(target_y),1000)).sum() + \
            l2_reg * (img**2).sum()
        loss.backward()
        with torch.no_grad():
            img += learning_rate * img.grad
        # model.zero_grad()
        img.grad.zero_()

        # Undo the random jitter
        img.data.copy_(jitter(img.data, -ox, -oy))

        # As regularizer, clamp and periodically blur the image
        for c in range(3):
            lo = float(-SQUEEZENET_MEAN[c] / SQUEEZENET_STD[c])
            hi = float((1.0 - SQUEEZENET_MEAN[c]) / SQUEEZENET_STD[c])
            img.data[:, c].clamp_(min=lo, max=hi)
        if t % blur_every == 0:
            blur_image(img.data, sigma=0.5)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            plt.imshow(deprocess(img.data.clone().cpu()))
            class_name = class_names[target_y]
            plt.title('%s\nIteration %d / %d' %
                      (class_name, t + 1, num_iterations))
            plt.gcf().set_size_inches(4, 4)
            plt.axis('off')
            plt.show()

    return deprocess(img.data.cpu())


if __name__ == "__main__":
    model = torchvision.models.squeezenet1_1(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    from data_utils import load_imagenet_val
    X, y, class_names = load_imagenet_val(num=5)
    #############################################################################
    # saliency map                                                            #
    #############################################################################
    # plt.figure(figsize=(12, 6))
    # for i in range(5):
    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(X[i])
    #     plt.title(class_names[y[i]])
    #     plt.axis('off')
    # plt.gcf().tight_layout()
    # plt.show()

    # show_saliency_maps(X, y)
    
    #############################################################################
    # fool image                                                                #
    #############################################################################
    # idx = 0
    # target_y = 6

    # X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    # X_fooling = make_fooling_image(X_tensor[idx:idx + 1], target_y, model)

    # scores = model(X_fooling)
    # assert target_y == scores.data.max(
    #     1)[1][0].item(), 'The model is not fooled!'

    # X_fooling_np = deprocess(X_fooling.clone())
    # X_fooling_np = np.asarray(X_fooling_np).astype(np.uint8)

    # plt.subplot(1, 4, 1)
    # plt.imshow(X[idx])
    # plt.title(class_names[y[idx]])
    # plt.axis('off')

    # plt.subplot(1, 4, 2)
    # plt.imshow(X_fooling_np)
    # plt.title(class_names[target_y])
    # plt.axis('off')

    # plt.subplot(1, 4, 3)
    # X_pre = preprocess(Image.fromarray(X[idx]))
    # diff = np.asarray(deprocess(X_fooling - X_pre, should_rescale=False))
    # plt.imshow(diff)
    # plt.title('Difference')
    # plt.axis('off')

    # plt.subplot(1, 4, 4)
    # diff = np.asarray(deprocess(10 * (X_fooling - X_pre),
    #                             should_rescale=False))
    # plt.imshow(diff)
    # plt.title('Magnified difference (10x)')
    # plt.axis('off')

    # plt.gcf().set_size_inches(12, 5)
    # plt.show()

    #############################################################################
    # Class visualization                                                       #
    #############################################################################
    dtype = torch.FloatTensor
    # dtype = torch.cuda.FloatTensor # Uncomment this to use GPU
    model.type(dtype)


    # target_y = 78 # Tick
    # target_y = 187 # Yorkshire Terrier
    # target_y = 683 # Oboe
    # target_y = 366 # Gorilla
    # target_y = 604 # Hourglass
    target_y = np.random.randint(1000)
    print(class_names[target_y])
    X = create_class_visualization(target_y, model, dtype)