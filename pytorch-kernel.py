import torch.nn as nn
from torch.nn.utils import weight_norm


def print_linear():
    layer = nn.Linear(input_size, hidden_size)
    layer.weight.data.fill_(5)

    wn_layer = weight_norm(layer)
    print(wn_layer.weight)
    print(wn_layer.weight_g)
    print(wn_layer.weight_v)


def print_conv():
    conv = nn.Conv2d(1, 2, 3)
    conv.weight.data.fill_(8)
    print(conv.weight)

    wn_conv = weight_norm(conv)
    print(wn_conv.weight)
    print(wn_conv.weight_g)
    print(wn_conv.weight_v)


if __name__ == "__main__":
    input_size = 3
    hidden_size = 3

    # print_conv()
    print_linear()
