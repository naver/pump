# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

import pdb; bb = pdb.set_trace
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import DataParallel

from .common import todevice


class Trainer (nn.Module):
    """ Helper class to train a deep network.
        Overload this class `forward_backward` for your actual needs.
    
    Usage: 
        train = Trainer(net, loss, optimizer)
        for epoch in range(n_epochs):
            train()
    """
    def __init__(self, net, loss, optimizer, epoch=0):
        super().__init__()
        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        self.epoch = epoch

    @property
    def device(self):
        return next(self.net.parameters()).device

    @property
    def model(self):
        return self.net.module if isinstance(self.net, DataParallel) else self.net

    def distribute(self):
        self.net = DataParallel(self.net) # DataDistributed not implemented yet

    def __call__(self, data_loader):
        print(f'>> Training (epoch {self.epoch} --> {self.epoch+1})')
        self.net.train()

        stats = defaultdict(list)

        for batch in tqdm(data_loader):
            batch = todevice(batch, self.device)
            
            # compute gradient and do model update
            self.optimizer.zero_grad()
            details = self.forward_backward(batch)
            self.optimizer.step()

            for key, val in details.items():
                stats[key].append( val )

        self.epoch += 1

        print("   Summary of losses during this epoch:")
        for loss_name, vals in stats.items():
            N = 1 + len(vals)//10
            print(f"    - {loss_name:10}: {avg(vals[:N]):.3f} --> {avg(vals[-N:]):.3f} (avg: {avg(vals):.3f})")

    def forward_backward(self, inputs):
        raise NotImplementedError()

    def save(self, path):
        print(f"\n>> Saving model to {path}")

        data = {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'loss': self.loss.state_dict(),
                'epoch': self.epoch}

        torch.save(data, open(path,'wb'))

    def load(self, path, resume=True):
        print(f">> Loading weights from {path} ...")
        checkpoint = torch.load(path, map_location='cpu')
        assert isinstance(checkpoint, dict)

        self.net.load_state_dict(checkpoint['model'])
        if resume:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.loss.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch']
            print(f"   Resuming training at Epoch {self.epoch}!")


def get_loss( loss ):
    """ returns a tuple (loss, dictionary of loss details)
    """
    assert isinstance(loss, dict)
    grads = None

    k,l = next(iter(loss.items())) # first item is assumed to be the main loss
    if isinstance(l, tuple):
        l, grads = l
        loss[k] = l

    return (l, grads), {k:float(v) for k,v in loss.items()}


def backward( loss ):
    if isinstance(loss, tuple):
        loss, grads = loss
    else:
        loss, grads = (loss, None)

    assert loss == loss, 'loss is NaN'

    if grads is None:
        loss.backward()
    else:
        # dictionary of separate subgraphs
        for var,grad in grads:
             var.backward(grad)
    return float(loss)


def avg( lis ):
    return sum(lis) / len(lis)
