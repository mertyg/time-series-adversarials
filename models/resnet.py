import torch.nn as nn


class TSResNetBlock(nn.Module):
    def __init__(self, in_features, n_featuremaps):
        super(TSResNetBlock, self).__init__()
        self.in_features = in_features
        self.n_featuremaps = n_featuremaps
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=n_featuremaps,
                               kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(num_features=n_featuremaps)
        self.conv2 = nn.Conv1d(in_channels=n_featuremaps, out_channels=n_featuremaps,
                               kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(num_features=n_featuremaps)

        self.conv3 = nn.Conv1d(in_channels=n_featuremaps, out_channels=n_featuremaps,
                               kernel_size=3, padding=2)
        self.bn3 = nn.BatchNorm1d(num_features=n_featuremaps)
        self.relu = nn.ReLU()
        self.sc = nn.Conv1d(in_channels=in_features, out_channels=n_featuremaps,
                               kernel_size=1, padding=1)
        self.bnsc = nn.BatchNorm1d(num_features=n_featuremaps)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        shortcut = self.bnsc(self.sc(x))
        out = self.relu(shortcut+out)
        return out


class TSResNet(nn.Module):
    def __init__(self, args, loader):
        super(TSResNet, self).__init__()
        dataset = loader.dataset
        self.block1 = TSResNetBlock(in_features=dataset.n_variates, n_featuremaps=64)
        self.block2 = TSResNetBlock(in_features=64, n_featuremaps=128)
        self.block3 = TSResNetBlock(in_features=128, n_featuremaps=128)
        self.avgpool = nn.AvgPool1d(kernel_size=dataset.input_size-2)
        self.final = nn.Linear(in_features=128, out_features=dataset.output_size)
        self.features = nn.Sequential(self.block1, self.block2, self.block3)

    def forward(self, x):
        out = x
        out = out.view(out.shape[0], out.shape[2], out.shape[1])
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.avgpool(out).squeeze(dim=2)
        return self.final(out)
