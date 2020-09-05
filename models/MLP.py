import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, args, loader):
        dataset = loader.dataset
        super(MLP, self).__init__()
        print("Input size: ", dataset.input_size)
        self.fc1 = nn.Linear(dataset.input_size, 500)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.dropout = nn.Dropout(0.2)
        self.drop1 = nn.Dropout(0.1)
        self.drop3 = nn.Dropout(0.3)
        self.final_layer = nn.Linear(500, dataset.output_size)

    def forward(self, x, with_intermediate=False):
        activations = {}
        out = self.drop1(x.view(x.shape[0], -1))
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.drop3(self.relu(self.fc3(out)))
        out = self.final_layer(out)
        if with_intermediate:
            return out, activations
        return out


class DummyNet(nn.Module):
    def __init__(self, args, loader):
        dataset = loader.dataset
        super(DummyNet, self).__init__()
        self.fc1 = nn.Linear(dataset.input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.final_layer = nn.Linear(64, dataset.output_size)

    def forward(self, x, with_intermediate=False):
        out = x
        out = out.view(out.shape[0], -1)
        activations = []
        out = self.relu(self.fc1(out))
        activations.append(out)
        out = self.relu(self.fc2(out))
        activations.append(out)
        out = self.final_layer(out)

        if with_intermediate:
            return out, activations
        return out
