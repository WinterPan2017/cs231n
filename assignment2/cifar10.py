'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 
Author: Winter
Email: 837950571@qq.com
Date: 2020-12-07 14:56:58
LastEditTime: 2020-12-12 00:10:19
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as dset
from torch.utils.data import DataLoader, sampler
import matplotlib.pyplot as plt

device = torch.device('cpu')


def load_data(NUM_TRAIN=49000):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    cifar10_train = dset.CIFAR10('cs231n/datasets',
                                 train=True,
                                 download=True,
                                 transform=transform)
    loader_train = DataLoader(cifar10_train,
                              batch_size=64,
                              sampler=sampler.SubsetRandomSampler(
                                  range(NUM_TRAIN)))
    cifar10_val = dset.CIFAR10('cs231n/datasets',
                               train=True,
                               download=True,
                               transform=transform)
    loader_val = DataLoader(cifar10_val,
                            batch_size=64,
                            sampler=sampler.SubsetRandomSampler(
                                range(NUM_TRAIN, 50000)))
    cifar10_test = dset.CIFAR10('cs231n/datasets',
                                train=False,
                                download=True,
                                transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=64)
    return loader_train, loader_val, loader_test


class MyConvNet(nn.Module):
    """
    Conv -> relu -> Conv -> relu -> pool -> affine -> relu -> affine -> softmax

    input : (N, 3, 32, 32)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn10 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        # nn.init.kaiming_normal_(self.sn1.weight)
        # nn.init.kaiming_normal_(self.sn2.weight)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.kaiming_normal_(self.conv8.weight)
        nn.init.kaiming_normal_(self.conv9.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.dropout1(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.dropout2(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)

        x = self.dropout3(x)

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.bn10(x)
        x = F.relu(x)
        x = self.dropout4(x)
        scores = self.fc2(x)
        return scores


def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct, num_samples = 0, 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size()[0]
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, acc))
        return acc


def train(model, optimizer, loaders, epochs=1, print_every=100, debug=False):
    loader_train, loader_val, loader_test = loaders
    model = model.to(device=device)
    epochs_history = []
    train_loss_history = []
    train_acc_history = []
    val_acc_history = []
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # set model to training mode
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if t % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' %
                      (e, t, loss.item()))
                va = check_accuracy(loader_val, model)
                print()
                if debug:
                    ta = check_accuracy(loader_train, model)
                    train_acc_history.append(ta)
                train_loss_history.append(loss.item())
                val_acc_history.append(va)
    return train_loss_history, train_acc_history, val_acc_history


def visualize(history):
    train_loss_history, train_acc_history, val_acc_history = history
    plt.subplot(2, 1, 1)
    x = np.arange(1, len(train_acc_history) + 1, 1)
    plt.title('loss on training set')
    plt.plot(x, train_acc_history, label='loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('accuracy on training set and validation set ')
    plt.plot(x, train_acc_history, label='train acc')
    plt.plot(x, val_acc_history, label='val acc')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    loaders = load_data()

    model = MyConvNet()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
    history = train(model, optimizer, loaders, epochs=20, debug=False)
    print('Finally')
    train_acc = check_accuracy(loaders[0], model)
    val_acc = check_accuracy(loaders[1], model)
    torch.save(model, 'model.pkl')

    # best_model = None
    # best_val = 0
    # best = None
    # res = []
    # nors = ['bn', 'sn']
    # lrs = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # weight_decays = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # for lr in lrs:
    #     for weight_decay in weight_decays:
    # for nor in nors:
    #     model = MyConvNet(normalize=nor)
    #     optimizer = optim.Adam(model.parameters(),
    #                                lr=5e-4,
    #                                weight_decay=5e-4)
    #     history = train(model, optimizer, loaders, epochs=10, debug=False)
    #     print('Finally')
    #     train_acc = check_accuracy(loaders[0], model)
    #     val_acc = check_accuracy(loaders[1], model)
    #     res.append((nor, train_acc, val_acc))
    #     if best_val < val_acc:
    #         best_model = model
    #         best_val = val_acc
    #         best = (nor, train_acc, val_acc)

    # for nor, train_acc, val_acc in res:
    #     print('%s, train acc=%.2f, val_acc=%.2f' %
    #           (nor, train_acc, val_acc))
