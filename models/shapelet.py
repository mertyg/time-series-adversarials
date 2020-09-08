import torch.nn as nn
import torch
import numpy as np


def pairwise_dist(x, y):
  x_norm = (x.norm(dim=2)[:, :, None])
  y_t = y.permute(0, 2, 1).contiguous()
  y_norm = (y.norm(dim=2)[:, None])
  y_t = torch.cat([y_t] * x.shape[0], dim=0)
  dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
  return torch.clamp(dist, 0.0, np.inf)


class ShapeletNet(nn.Module):
    def __init__(self, args, loader, bag_ratio=0.2):
        super(ShapeletNet, self).__init__()
        self.n_shapelets = 0
        self.bag_ratio = bag_ratio
        self.bag_size = int(bag_ratio * loader.dataset.input_size)
        self.n_variates = loader.dataset.n_variates
        print("N_variates: ", self.n_variates)
        self.shapelets = nn.Parameter(self.init_shapelets(args, loader))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.n_shapelets*self.n_variates, loader.dataset.output_size)
        #self.fc1 = nn.Linear(5*self.n_shapelets, 256)
        #self.fc2 = nn.Linear(256, 512)
        #self.fc3 = nn.Linear(512, 512)
        #self.fc4 = nn.Linear(512, loader.dataset.output_size)
        self.dropout = nn.Dropout(p=0.2)

    def init_shapelets(self, args, loader, n_shapelets=5):
        n_variates = loader.dataset.n_variates
        shapelets = 1e-2 * torch.randn((1, n_shapelets, n_variates, self.bag_size, 1))
        self.n_shapelets = n_shapelets
        return shapelets

    def convert_to_bags(self, data):
        bag_size = self.bag_size
        shift_size = self.bag_size // 2
        bags = []
        window_marker = 0
        while window_marker + bag_size < data.shape[2]:
            fragment = data[:, :, window_marker: window_marker+bag_size].unsqueeze(-1)
            bags.append(fragment)
            window_marker += shift_size
        bags = torch.cat(bags, dim=3)
        return bags

    def get_distance_features(self, input):
        # Input : batch_size x n_variates x bag_size x n_bags
        # Shapelets : n_shapelets x n_variates x bag_size
        # Return : batch_size x n_variates x n_shapelets
        input = input.view(input.shape[0], 1, self.n_variates, self.bag_size, input.shape[3])
        shapelets = self.shapelets
        # Batch_size x N_shapelets x N_variates x bag_size x N_bags
        diff = torch.norm(input-shapelets, p=2, dim=3)
        #dist_features = diff.view(diff.shape[0], -1)
        dist_features = diff
        min_features = diff.min(dim=-1)[0]
        #max_features = diff.max(dim=-1)[0]
        #dist_features = torch.cat([min_features, max_features], dim=1)
        dist_features = min_features
        # Batch_size x 2N_shapelets x N_variates
        dist_features = dist_features.view(dist_features.shape[0], -1)
        return dist_features

    def forward(self, x):
        # X is (batch_size, input_size, n_variates)
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        x = self.convert_to_bags(x) # batch_size x n_variates x bag_size x n_bags
        dist_features = self.get_distance_features(x) # batch_size x n_variates x n_shapelets
        return self.fc1(dist_features)
        #out = self.dropout(self.relu(self.fc1(dist_features)))
        #out = self.dropout(self.relu(self.fc2(out)))
        #out = self.dropout(self.relu(self.fc3(out)))
        #out = self.fc4(out)
        #return out

