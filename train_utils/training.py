import torch.nn.functional as F
from robustness.tools.helpers import AverageMeter, accuracy
import torch
import torch.nn as nn
import foolbox as fb
from tqdm import tqdm
import numpy as np


def train_step(args, model, loader, optimizer, batch_wrap = lambda x : x):
    # Loop adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py
    model.train()

    for (data, target) in batch_wrap(loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        if args.distance_loss:
            if args.model != "ShapeletNet":
                raise NotImplementedError()
            output, distances = model(data, return_dist=args.distance_loss)
            loss = F.nll_loss(F.log_softmax(output), target.long())
            dist_loss = distances.mean()
            loss = loss+dist_loss
        else:
            output = model(data)
            loss = F.nll_loss(F.log_softmax(output), target.long())
        loss.backward()
        optimizer.step()


def eval(model, loader, args, batch_wrap=lambda x: x):

    model.eval()
    losses = AverageMeter()
    if args.distance_loss and args.model=="ShapeletNet":
        dist_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for (inputs, targets) in batch_wrap(loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            N = targets.size(0)
            if args.distance_loss and args.model=="ShapeletNet":
                outputs, dists = model(inputs, args.distance_loss)
                dist_loss.update(dists.mean(), N)
            else:
                outputs = model(inputs)
            loss = F.nll_loss(F.log_softmax(outputs), targets)

            if loader.dataset.output_size < 5:
                prec1 = accuracy(outputs, targets, topk=[1])

            else:
                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                top5.update(prec5[0], N)
            losses.update(loss.item(), N)
            top1.update(prec1[0], N)

    top1_acc = top1.avg
    loss = losses.avg
    if loader.dataset.output_size < 5:
        report = {"Top1": top1_acc.item(), "Loss": loss}
    else:
        top5_acc = top5.avg
        report = {"Top1": top1_acc.item(), "Top5": top5_acc.item(), "Loss": loss}

    if args.distance_loss:
        report.update({"DistanceLoss": dist_loss.avg.item()})
    return report


def adversarial_eval(attack_fn, model, loader, epsilons, args):
    model.eval()
    fmodel = fb.PyTorchModel(model, bounds=(-np.inf, np.inf))
    robust_accuracies = {}

    for epsilon in epsilons:
        robust_accuracies[epsilon] = AverageMeter()

    t_loader = tqdm(loader)
    for (data, target) in t_loader:
        data, target = data.to(args.device), target.to(args.device)
        t_loader.set_description(" ".join([f"{k}: {v.avg}" for k, v in robust_accuracies.items()]))
        _, advs, success = attack_fn(fmodel, data, target, epsilons=epsilons)

        #ts1 = advs[1].detach().cpu().numpy()
        #ts2 = data.detach().cpu().numpy()
        #import matplotlib
        #matplotlib.use("TkAgg")
        #import matplotlib.pyplot as plt
        #fig, axs = plt.subplots(ts1.shape[0], 1, figsize=(10, 80))
        #for i in range(ts1.shape[0]):
        #    axs[i].plot(np.arange(ts1[i].shape[0]), ts1[i], label="adversarial")
        #    axs[i].plot(np.arange(ts2[i].shape[0]), ts2[i], label="regular")
        #    axs[i].legend()
        #fig.savefig("./comparison.png")
        #exit(1)
        robust_accuracy = 1 - success.float().mean(axis=-1)
        N = target.size(0)
        for eps, accuracy in zip(epsilons, robust_accuracy):
            robust_accuracies[eps].update(accuracy.item(), N)
    for k, v in robust_accuracies.items():
        robust_accuracies[k] = v.avg
    return robust_accuracies


