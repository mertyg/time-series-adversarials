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
        self.fc1 = nn.Linear(2*self.n_shapelets*self.n_variates, loader.dataset.output_size)
        self.bn = nn.BatchNorm1d(num_features=2*self.n_shapelets*self.n_variates)

        self.softmax = nn.Softmax(dim=-1)

    def init_shapelets(self, args, loader, n_shapelets=5):
        n_variates = loader.dataset.n_variates
        shapelets = 0.1 * torch.randn((1, n_shapelets, n_variates, self.bag_size, 1))
        self.n_shapelets = n_shapelets
        return shapelets

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

    def get_distance_features(self, input):
        # Input : batch_size x n_variates x bag_size x n_bags
        # Shapelets : n_shapelets x n_variates x bag_size
        # Return : batch_size x n_variates x n_shapelets
        input = input.view(input.shape[0], 1, self.n_variates, self.bag_size, input.shape[3])
        shapelets = self.shapelets
        # Batch_size x N_shapelets x N_variates x bag_size x N_bags
        diff = torch.norm(input-shapelets, p=2, dim=3)
        return diff

    def get_similarity_shapelets(self):
        shapelets = self.shapelets.view(self.n_shapelets, self.bag_size)
        #prod = torch.einsum("ik, jk -> ij", shapelets, shapelets)
        #norms = torch.norm(shapelets, dim=1)
        #norms_mult = torch.einsum("i, j-> ij", norms, norms)
        #cos_sim = prod/norms_mult
        mean_shapelet = shapelets.mean(dim=0, keepdim=True)
        dist = torch.norm(mean_shapelet-shapelets, p=2)
        return -dist

    def forward(self, x, return_dist=False):
        # X is (batch_size, input_size, n_variates)
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        x = self.convert_to_bags(x) # batch_size x n_variates x bag_size x n_bags
        diffs = self.get_distance_features(x) # batch_size x n_variates x n_shapelets x n_bags
        dist_min = diffs.min(dim=-1)[0].view(diffs.shape[0], -1)
        dist_avg = diffs.mean(dim=-1).view(diffs.shape[0], -1)
        dist_features = torch.cat([dist_min, dist_avg], dim=1)
        dist_features = self.bn(dist_features)
        out = self.fc1(dist_features)
        if return_dist:
            return out, self.get_similarity_shapelets()+dist_min.mean()
        return out
