'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 

             Finished features.ipynb here.
             
Author: Winter
Email: 837950571@qq.com
Date: 2020-11-12 14:30:10
LastEditTime: 2020-12-20 15:56:10
'''
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from data_utils import load_CIFAR10
from linear_svm import LinearSVM
from neural_network import TwoLayerNet


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def hog_feature(im):
    """
    Compute Histogram of Gradient (HOG) feature for an image

      Parameters:
        im : an input grayscale or rgb image

      Returns:
        feat: Histogram of Gradient (HOG) feature

    """
    # convert rgb to grayscale if needed
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape  # image size
    orientations = 9  # number of gradient bins
    cx, cy = (8, 8)  # pixels per cell

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # compute gradient on x-direction
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # compute gradient on y-direction
    grad_mag = np.sqrt(gx**2 + gy**2)  # gradient magnitude
    grad_ori = np.arctan2(
        gx, (gy + 1e-15)) * (180 / np.pi) + 90  # gradient orientation

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        temp_ori = np.where(grad_ori < (180 / orientations) * (i + 1),
                            grad_ori, 0)
        temp_ori = np.where(temp_ori >= (180 / orientations) * i, temp_ori, 0)
        temp_mag = np.where(temp_ori > 0, grad_mag, 0)
        orientation_histogram[:, :, i] = uniform_filter(
            temp_mag, size=(cx, cy))[round(cx / 2)::cx,
                                     round(cy / 2)::cy].T

    return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    Compute color histogram for an image using hue.

    Inputs:
    - im: H x W x C array of pixel data for an RGB image.
    - nbin: Number of histogram bins. (default: 10)
    - xmin: Minimum pixel value (default: 0)
    - xmax: Maximum pixel value (default: 255)
    - normalized: Whether to normalize the histogram (default: True)

    Returns:
      1D vector of length nbin giving the color histogram over the hue of the
      input image.
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0],
                                     bins=bins,
                                     density=normalized)
    imhist = imhist * np.diff(bin_edges)

    return imhist


def extract_features(imgs, feature_fns, verbose=False):
    """
    Given pixel data for images and several feature functions that can operate on
    single images, apply all feature functions to all images, concatenating the
    feature vectors for each image and storing the features for all images in
    a single matrix.

    Inputs:
    - imgs: N x H X W X C array of pixel data for N images.
    - feature_fns: List of k feature functions. The ith feature function should
      take as input an H x W x D array and return a (one-dimensional) array of
      length F_i.
    - verbose: Boolean; if true, print progress.

    Returns:
    An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
    of all features for a single image.
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(
            feats.shape) == 1, 'Feature functions must be one-dimensional'
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # Extract features for the rest of the images.
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 999:
            print('Done extracting features for %d / %d images' %
                  (i + 1, num_images))

    return imgs_features


if __name__ == "__main__":
    num_training = 49000
    num_validation = 1000
    num_test = 1000

    X_train, y_train, X_test, y_test = load_CIFAR10()
    X_val = X_train[num_training:num_training + num_validation, ...]
    y_val = y_train[num_training:num_training + num_validation]
    X_train = X_train[:num_training, ...]
    y_train = y_train[:num_training]
    X_test = X_test[:num_test, ...]
    y_test = y_test[:num_test]

    # process raw image to get the features of HOG and HSV
    num_color_bins = 10
    feature_fns = [
        hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)
    ]
    X_train_feats = extract_features(X_train, feature_fns, verbose=True)
    X_val_feats = extract_features(X_val, feature_fns)
    X_test_feats = extract_features(X_test, feature_fns)

    mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
    X_train_feats -= mean_feat
    X_val_feats -= mean_feat
    X_test_feats -= mean_feat

    # Preprocessing: Divide by standard deviation.
    # This ensures that each feature has roughly the same scale.
    std_feat = np.std(X_train_feats, axis=0, keepdims=True)
    X_train_feats /= std_feat
    X_val_feats /= std_feat
    X_test_feats /= std_feat

    # Preprocessing: Add a bias dimension
    X_train_feats = np.hstack(
        [X_train_feats, np.ones((X_train_feats.shape[0], 1))])
    X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
    X_test_feats = np.hstack(
        [X_test_feats, np.ones((X_test_feats.shape[0], 1))])

    # Use the validation set to tune the learning rate and regularization strength
    learning_rates = [1e-9, 1e-8, 1e-7]
    regularization_strengths = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    results = {}
    best_val = -1
    best_svm = None

    for learning_rate in learning_rates:
        for regularization_strength in regularization_strengths:
            model = LinearSVM()
            model.train(X_train_feats,
                        y_train,
                        learning_rate == learning_rate,
                        reg=regularization_strength,
                        num_iters=2000)
            train_acc = np.mean(model.predict(X_train_feats) == y_train)
            val_acc = np.mean(model.predict(X_val_feats) == y_val)
            results[(learning_rate, regularization_strength)] = (train_acc,
                                                                 val_acc)
            if val_acc > best_val:
                best_val = val_acc
                best_svm = model

    # Print out results.
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' %
              (lr, reg, train_accuracy, val_accuracy))

    print('best validation accuracy achieved during cross-validation: %f' %
          best_val)
    # Evaluate your trained SVM on the test set
    y_test_pred = best_svm.predict(X_test_feats)
    test_accuracy = np.mean(y_test == y_test_pred)
    print(test_accuracy)

    # # Preprocessing: Remove the bias dimension
    # X_train_feats = X_train_feats[:, :-1]
    # X_val_feats = X_val_feats[:, :-1]
    # X_test_feats = X_test_feats[:, :-1]

    # input_dim = X_train_feats.shape[1]
    # hidden_dim = 500
    # num_classes = 10

    # # net = TwoLayerNet(input_dim, hidden_dim, num_classes)
    # best_net = None
    # best_val = -1
    # results = {}
    # learning_rates = [10, 5, 2, 1, 1e-1]  # 1
    # regs = [1e-3, 1e-4, 1e-5]  # 1e-3
    # for lr in learning_rates:
    #     for reg in regs:
    #         net = TwoLayerNet(input_dim, hidden_dim, num_classes)
    #         net.train(
    #             X_train_feats,
    #             y_train,
    #             X_val_feats,
    #             y_val,
    #             learning_rate=lr,
    #             #   learning_rate_decay=0.95,
    #             reg=reg,
    #             num_iters=5000)
    #         acc_train = np.mean(net.predict(X_train_feats) == y_train)
    #         acc_val = np.mean(net.predict(X_val_feats) == y_val)
    #         results[(lr, reg)] = (acc_train, acc_val)
    #         if best_val < acc_val:
    #             best_val = acc_val
    #             best_net = net

    # for lr, reg in sorted(results):
    #     train_accuracy, val_accuracy = results[(lr, reg)]
    #     print('lr %e reg %e train accuracy: %f val accuracy: %f' %
    #           (lr, reg, train_accuracy, val_accuracy))

    # test_acc = (best_net.predict(X_test_feats) == y_test).mean()
    # print(test_acc)