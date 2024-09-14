"""
Solver Class.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import time
import copy
import pathlib
import os

import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data import Imbalanced_CIFAR10
from models import TwoLayerNet, VanillaCNN, MyModel, resnet32
from losses import FocalLoss, reweight


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Solver(object):
    def __init__(self, **kwargs):
        self.path_prefix = kwargs.pop("path_prefix", ".")
        self.imbalance = kwargs.pop("imbalance", "regular")
        self.batch_size = kwargs.pop("batch_size", 128)
        self.model_type = kwargs.pop("model", "TwoLayerNet")
        self.device = kwargs.pop("device", "cpu")
        self.loss_type = kwargs.pop("loss_type", "CE")
        self.lr = kwargs.pop("learning_rate", 0.0001)
        self.momentum = kwargs.pop("momentum", 0.9)
        self.reg = kwargs.pop("reg", 0.0005)
        self.beta = kwargs.pop("beta", 0.9999)
        self.gamma = kwargs.pop("gamma", 1.0)
        self.steps = kwargs.pop("steps", [6, 8])
        self.epochs = kwargs.pop("epochs", 10)
        self.warmup = kwargs.pop("warmup", 0)
        self.save_best = kwargs.pop("save_best", True)

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        if self.imbalance == "regular":
            train_dataset = torchvision.datasets.CIFAR10(
                root=os.path.join(self.path_prefix, "data", "cifar10"),
                train=True,
                download=True,
                transform=transform_train,
            )
        else:
            train_dataset = Imbalanced_CIFAR10(
                root=os.path.join(self.path_prefix, "data", "cifar10"),
                download=True,
                transform=transform_train,
            )
            cls_num_list = train_dataset.get_cls_num_list()
            if self.loss_type == "Focal":
                per_cls_weights = reweight(cls_num_list, beta=self.beta)
                per_cls_weights = per_cls_weights.to(self.device)
            else:
                per_cls_weights = None

        # Normalize the test set same as training set without augmentation
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=100, shuffle=False, num_workers=2
        )

        if self.model_type == "TwoLayerNet":
            self.model = TwoLayerNet(3072, 256, 10)
        elif self.model_type == "VanillaCNN":
            self.model = VanillaCNN()
        elif self.model_type == "MyModel":
            self.model = MyModel()
        elif self.model_type == "ResNet-32":
            self.model = resnet32()

        print(self.model)
        self.model = self.model.to(self.device)

        if self.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = FocalLoss(weight=per_cls_weights, gamma=self.gamma)

        self.criterion.to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.lr,
            momentum=self.momentum,
            weight_decay=self.reg,
        )

        self._reset()

    def _reset(self):
        self.best = 0.0
        self.best_cm = None
        self.best_model = None

    def train(self):
        for epoch in range(self.epochs):
            self._adjust_learning_rate(epoch)

            # train loop
            self._train_step(epoch)

            # validation loop
            acc, cm = self._evaluate(epoch)

            if acc > self.best:
                self.best = acc
                self.best_cm = cm
                self.best_model = copy.deepcopy(self.model)

        print("Best Prec @1 Acccuracy: {:.4f}".format(self.best))
        per_cls_acc = self.best_cm.diag().detach().numpy().tolist()
        for i, acc_i in enumerate(per_cls_acc):
            print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

        if self.save_best:
            basedir = pathlib.Path(__file__).parent.resolve()
            torch.save(
                self.best_model.state_dict(),
                str(basedir) + "/checkpoints/" + self.model_type.lower() + ".pth",
            )

    def _train_step(self, epoch):
        iter_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        self.model.train()

        for idx, (data, target) in enumerate(self.train_loader):
            start = time.time()

            data = data.to(self.device)
            target = target.to(self.device)

            out, loss = self._compute_loss_update_params(data, target)

            batch_acc = self._check_accuracy(out, target)

            losses.update(loss.item(), out.shape[0])
            acc.update(batch_acc, out.shape[0])

            iter_time.update(time.time() - start)
            if idx % 10 == 0:
                print(
                    (
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t"
                    ).format(
                        epoch,
                        idx,
                        len(self.train_loader),
                        iter_time=iter_time,
                        loss=losses,
                        top1=acc,
                    )
                )

    def _evaluate(self, epoch):
        iter_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        num_class = 10
        cm = torch.zeros(num_class, num_class)
        self.model.eval()

        # evaluation loop
        for idx, (data, target) in enumerate(self.val_loader):
            start = time.time()

            data = data.to(self.device)
            target = target.to(self.device)

            out, loss = self._compute_loss_update_params(data, target)

            batch_acc = self._check_accuracy(out, target)

            # update confusion matrix
            _, preds = torch.max(out, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

            losses.update(loss.item(), out.shape[0])
            acc.update(batch_acc, out.shape[0])

            iter_time.update(time.time() - start)
            if idx % 10 == 0:
                print(
                    (
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                    ).format(
                        epoch,
                        idx,
                        len(self.val_loader),
                        iter_time=iter_time,
                        loss=losses,
                        top1=acc,
                    )
                )
        cm = cm / cm.sum(1)
        per_cls_acc = cm.diag().detach().numpy().tolist()
        for i, acc_i in enumerate(per_cls_acc):
            print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

        print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
        return acc.avg, cm

    def _check_accuracy(self, output, target):
        """Computes the precision@k for the specified values of k"""
        batch_size = target.shape[0]

        _, pred = torch.max(output, dim=-1)

        correct = pred.eq(target).sum() * 1.0

        acc = correct / batch_size

        return acc

    def _compute_loss_update_params(self, data, target):
        output = None
        loss = None
        if self.model.training:
            #############################################################################
            # TODO: Complete the body of training loop                                  #
            #       1. forward data batch to the model                                  #
            #       2. Compute batch loss                                               #
            #       3. Compute gradients and update model parameters                    #
            #############################################################################
            pass
            #############################################################################
            #                              END OF YOUR CODE                             #
            #############################################################################
        else:
            #############################################################################
            # TODO: Complete the body of training loop                                  #
            #       HINT: torch.no_grad()                                               #
            #############################################################################
            pass
            #############################################################################
            #                              END OF YOUR CODE                             #
            #############################################################################

        return output, loss

    def _adjust_learning_rate(self, epoch):
        epoch += 1
        if epoch <= self.warmup:
            lr = self.lr * epoch / self.warmup
        elif epoch > self.steps[1]:
            lr = self.lr * 0.01
        elif epoch > self.steps[0]:
            lr = self.lr * 0.1
        else:
            lr = self.lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
