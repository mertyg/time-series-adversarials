import torch.nn as nn
import torch
import numpy as np


class ConvShapeletNet(nn.Module):
    def __init__(self, args, loader, bag_ratio=.15):
        super(ConvShapeletNet, self).__init__()
        self.n_shapelets = 0
        self.bag_ratio = bag_ratio
        self.bag_size = int(bag_ratio * loader.dataset.input_size)
        self.n_variates = loader.dataset.n_variates
        print("N_variates: ", self.n_variates)
        n_out = 10
        self.conv_shapelet = nn.Conv1d(in_channels=self.n_variates, out_channels=n_out, kernel_size=self.bag_size,
                                       stride=self.bag_size//4, bias=True, groups=self.n_variates)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8*n_out, loader.dataset.output_size)
        #self.fc2 = nn.Linear(3 * n_out, loader.dataset.output_size)
        self.conv_shapelet.weight = nn.Parameter(torch.ones_like(self.conv_shapelet.weight), requires_grad=False)
        self.pool = nn.MaxPool1d(kernel_size=3)

    def convert_to_bags(self, data):
        bag_size = self.bag_size
        shift_size = self.bag_size // 2
        bags = []
        window_marker = 0
        # Data: BatchSize x N_Variates X TS_length
        while window_marker + bag_size < data.shape[2]:
            fragment = data[:, :, window_marker: window_marker+bag_size].unsqueeze(-1)
            bags.append(fragment)
            window_marker += shift_size
        bags = torch.cat(bags, dim=3)
        return bags

    def forward(self, x):
        # X is (batch_size, input_size, n_variates)
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        #x = self.convert_to_bags(x) # batch_size x n_variates x bag_size x n_bags
        x = (self.conv_shapelet(x))**2
        x = self.pool(-x).view(x.shape[0], -1)
        #print(x.shape)
        #x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
