import torch.nn.functional as F
from robustness.tools.helpers import AverageMeter, accuracy
import torch
import torch.nn as nn
import foolbox as fb


def train_step(args, model, loader, optimizer, batch_wrap = lambda x : x):
    # Loop adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py
    model.train()

    for (data, target) in batch_wrap(loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output), target.long())
        loss.backward()
        optimizer.step()


def eval(model, loader, args, batch_wrap=lambda x: x):

    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for (inputs, targets) in batch_wrap(loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = F.nll_loss(F.log_softmax(outputs), targets)
            N = targets.size(0)

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

    return report


def adversarial_eval(model, loader, epsilons=[0.0, 0.001, 0.01, 0.1]):
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = fb.attacks.LinfPGD()
    robust_accuracies = {}

    for epsilon in epsilons:
        robust_accuracies[epsilon] = AverageMeter()

    for (data, target) in loader:
        _, advs, success = attack(fmodel, data, target, epsilons=epsilons)
        robust_accuracy = 1 - success.float().mean(axis=-1)
        N = target.size(0)
        for eps, accuracy in zip(epsilons, robust_accuracy):
            robust_accuracies[eps].update(accuracy.item(), N)

    return robust_accuracies


