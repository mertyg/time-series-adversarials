import torch.nn as nn
# https://github.com/wkentaro/pytorch-fcn


class FCN(nn.Module):
    def __init__(self, args, loader):
        dataset = loader.dataset
        super(FCN, self).__init__()
        print("Input size: ", dataset.input_size)
        print("N_Variates: ", dataset.n_variates)
        print("Output Size: ", dataset.output_size)

        self.conv1 = nn.Conv1d(in_channels=dataset.n_variates,
                               out_channels=128, kernel_size=8)

        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=128,
                               out_channels=256, kernel_size=5)

        self.bn2 = nn.BatchNorm1d(num_features=256)

        self.conv3 = nn.Conv1d(in_channels=256,
                               out_channels=128, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.avgpool = nn.AvgPool1d(kernel_size=dataset.input_size-15)
        self.final_layer = nn.Linear(128, dataset.output_size)

    def forward(self, x, with_intermediate=False):
        activations = {}
        out = x
        out = out.view(out.shape[0], out.shape[2], out.shape[1])
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.avgpool(out).squeeze(dim=2)
        out = self.final_layer(out)
        if with_intermediate:
            return out, activations
        return out
