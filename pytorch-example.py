"""
PyTorch Weight Norm Implementation

Collab:
https://colab.research.google.com/drive/19MY8wo2p-6V3rnfSPpPG-2BLruL4d4yQ
"""
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.utils import weight_norm
from matplotlib import pyplot as plt


class Net(nn.Module):
    """
    Standard conv net
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class WeightNormNet(nn.Module):
    """
    Conv net with weight normalization applied
    """
    def __init__(self):
        super(WeightNormNet, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(3, 6, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = weight_norm(nn.Conv2d(6, 16, 5))
        self.fc1 = weight_norm(nn.Linear(16 * 5 * 5, 120))
        self.fc2 = weight_norm(nn.Linear(120, 84))
        self.fc3 = weight_norm(nn.Linear(84, 10))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(network, opt):
    """
    Train network architecture with given optimizer
    :param network: `nn.Module` instance
    :param opt: Criterion optimizer
    :return: Numpy array of losses
    """
    running_loss_array = []

    for epoch in range(50):  # 50 epochs

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            opt.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()

            if i % 32 == 31:
                running_loss_array.append(running_loss / 32)
                running_loss = 0.0

    return running_loss_array


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=2)

    regular_net = Net().cuda()
    regular_optimizer = optim.SGD(regular_net.parameters(), lr=0.001,
                                  momentum=0.9)

    weightnorm_net = WeightNormNet().cuda()
    weightnorm_optimizer = optim.SGD(weightnorm_net.parameters(),
                                     lr=0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    regular_loss = train(regular_net, regular_optimizer)
    weightnorm_loss = train(weightnorm_net, weightnorm_optimizer)

    regular_loss = np.asarray(regular_loss)
    weightnorm_loss = np.asarray(weightnorm_loss)

    num_data = regular_loss.shape[0]
    plt.plot(np.linspace(0, num_data, num_data), weightnorm_loss,
             color='red', label='weightnorm')

    plt.plot(np.linspace(0, num_data, num_data), regular_loss,
             color='blue', label='regular parameterization')

    plt.legend()
    plt.show()
