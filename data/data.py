import torch
from tslearn.datasets import UCR_UEA_datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
import numpy as np
from sklearn import preprocessing
import os


class UCRDataset(Dataset):
    """UCR Time Series dataset."""
    def __init__(self, X, y, name, transform=None):
        if transform is None:
            transform = Compose([ToTensor()])
        self.transform = transform
        self.data = X.astype(np.float32)
        self.target = y
        self.__name__ = f"{name} Dataset"
        self.input_size = self.data.shape[1]
        max_class = np.max(self.target)
        self.output_size = int(max_class + 1)
        self.n_variates = self.data.shape[2]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.detach().cpu().numpy()
        X_batch = self.transform(self.data[idx]).squeeze(dim=0)
        y_batch = torch.tensor(self.target[idx])
        return X_batch, y_batch


def load_dataset(args):
    dataset_name = args.dataset.split("_")[1]
    print(f"Trying to fetch {dataset_name}")
    datasets = UCR_UEA_datasets()
    datasets._data_dir = os.path.join(args.base_dir, "dataset")

    if not os.path.isdir(datasets._data_dir):
        os.makedirs(datasets._data_dir)

    X_train, y_train, X_test, y_test = datasets.load_dataset(dataset_name)


    #min_x = np.min(X_train, axis=(0, 1), keepdims=True)
    #max_x = np.min(X_train, axis=(0, 1), keepdims=True)
    #X_train = (X_train-min_x)/max_x
    #X_test = (X_test-min_x)/max_x

    mu_train = np.mean(X_train, axis=1, keepdims=True)
    std_train = np.std(X_train, axis=1, keepdims=True)
    std_train[std_train == 0] = 1
    mu_test = np.mean(X_test, axis=1, keepdims=True)
    std_test = np.std(X_test, axis=1, keepdims=True)
    std_test[std_test == 0] = 1

    X_train = (X_train-mu_train)/std_train
    X_test = (X_test-mu_test)/std_test

    norm_mean = np.mean(X_train, axis=0, keepdims=True)
    norm_std = np.std(X_test, axis=0, keepdims=True)
    X_train = (X_train-norm_mean)/norm_std
    X_test = (X_test-norm_mean)/norm_std


    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate([y_train, y_test], axis=0))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    train_dataset = UCRDataset(X_train, y_train, dataset_name)
    test_dataset = UCRDataset(X_test, y_test, dataset_name)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
    return train_loader, test_loader

