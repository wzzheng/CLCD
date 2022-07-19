import torch
import torch.nn as nn
import math
from copy import deepcopy
import logging
import torch.distributed as dist 

from gedml.core.collectors import BaseCollector
from gedml.core.misc import utils
from gedml.core.models import BatchNormMLP

class SimSiamMultiCollector(BaseCollector):

    def __init__(
        self,
        reduction_size=[512, 128, 32],
        first_bn = True,
        last_bn = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs) 
        self.bn = nn.BatchNorm1d(reduction_size[0]) if first_bn else nn.Identity()

        self.num_levels = len(reduction_size) - 1
        self.reduction = nn.ModuleList([
            BatchNormMLP(reduction_size[i:i+2], [False], [True]) for i in range(self.num_levels)
        ])
        self.relu = nn.ReLU()
        self.recover = nn.ModuleList([
            BatchNormMLP([reduction_size[i+1], reduction_size[i]], [False], [last_bn]) for i in range(self.num_levels)
        ])
        print("using fined grained simsiam multi level collector")
    
    def forward(self, data, embeddings, labels):

        # split two streams
        N = embeddings.size(0)
        assert N % 2 == 0
        z1, z2 = embeddings[:N//2], embeddings[N//2:]
        
        # pass through batchnorm layer
        z1, z2 = self.bn(z1), self.bn(z2)
        # list of metric mats of different levels
        metric_mats = []
        reduced1, reduced2 = z1, z2
        for i in range(self.num_levels):
            tmp_reduced1 = self.reduction[i](self.relu(reduced1))
            recover1 = self.recover[i](self.relu(tmp_reduced1))

            tmp_reduced2 = self.reduction[i](self.relu(reduced2))
            recover2 = self.recover[i](self.relu(tmp_reduced2))

            metric_mat = 0.5 * (
                self.metric(recover1, reduced2.detach()) +
                self.metric(recover2, reduced1.detach())
            )
            metric_mats.append(metric_mat)

            reduced1 = tmp_reduced1
            reduced2 = tmp_reduced2
        
        return (
            metric_mats,
            labels.unsqueeze(1),
            labels.unsqueeze(0),
            False
        )
