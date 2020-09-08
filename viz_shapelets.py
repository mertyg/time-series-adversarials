from data.data import load_dataset
from models import get_model
from config import get_arg_parser
from train_utils import train_step, eval, adversarial_eval, save_ckpt, load_ckpt
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    train_loader, test_loader = load_dataset(args)
    model = get_model(args)(args, train_loader)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
    best = np.inf
    model, optimizer, best, current_epoch = load_ckpt(args, model, optimizer)

    shapelets = model.shapelets.detach()
    shapelets = shapelets.squeeze()
    print(shapelets.shape)
    fig, axs = plt.subplots(shapelets.shape[0], 1)
    for i in range(shapelets.shape[0]):
        axs[i].plot(np.linspace(0, shapelets.shape[1]-1, num=shapelets.shape[1]), shapelets[i])
    fig.savefig("./shapelets.png")
