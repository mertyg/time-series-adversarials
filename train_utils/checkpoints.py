import torch
import os
import torch.nn as nn

def save_ckpt(args, model, optimizer, epoch, train_res, val_res, c_type="latest"):
    ckpt_folder = os.path.join(args.base_dir, args.dataset, args.model)
    if not os.path.isdir(ckpt_folder):
        os.makedirs(ckpt_folder)
    ckpt_path = os.path.join(ckpt_folder, f"ckpt.{c_type}")
    state_dict = {}
    state_dict["model"] = model.state_dict()
    state_dict["optimizer"] = optimizer.state_dict()
    state_dict["epoch"] = epoch
    state_dict["train_loss"] = train_res["Loss"]
    state_dict["val_loss"] = val_res["Loss"]
    state_dict["train_Top1"] = train_res["Top1"]
    state_dict["val_Top1"] = val_res["Top1"]
    torch.save(state_dict, ckpt_path)


def load_ckpt(args, model: nn.Module, optimizer):
    ckpt_folder = os.path.join(args.base_dir, args.dataset, args.model)
    if not os.path.isdir(ckpt_folder):
        os.makedirs(ckpt_folder)
    ckpt_path = os.path.join(ckpt_folder, "ckpt.latest")
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    best = state_dict["train_loss"]
    epoch = state_dict["epoch"]
    return model, optimizer, best, epoch
