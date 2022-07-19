import torch
import torch.nn as nn
import math
from copy import deepcopy
import logging
import torch.distributed as dist 

from gedml.core.collectors import BaseCollector
from gedml.core.misc import utils
from gedml.core.models import BatchNormMLP
from gedml.core.models.mlp import BatchNormLayer

class MultiBranchCollector(BaseCollector):

    def __init__(
        self,
        reduction_size=[512, 128, 32],
        first_bn = True,
        last_bn = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs) 
        self.bn = nn.BatchNorm1d(reduction_size[0], affine=False) if first_bn else nn.Identity()

        self.num_branches = len(reduction_size) - 1
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                BatchNormLayer(reduction_size[0], reduction_size[i+1], True, True, False),
                BatchNormLayer(reduction_size[i+1], reduction_size[0], False, last_bn, True)
            ) for i in range(self.num_branches)
        ])
    
    def forward(self, data, embeddings, labels):

        # split two streams
        N = embeddings.size(0)
        assert N % 2 == 0
        z1, z2 = embeddings[:N//2], embeddings[N//2:]
        
        # pass through batchnorm layer
        z1, z2 = self.bn(z1), self.bn(z2)
        # list of metric mats of different levels
        metric_mats = []
        for i in range(self.num_branches):
            recover1 = self.heads[i](z1)

            recover2 = self.heads[i](z2)

            metric_mat = 0.5 * (
                self.metric(recover1, z2.detach()) +
                self.metric(recover2, z1.detach())
            )
            metric_mats.append(metric_mat)
        
        return (
            metric_mats,
            labels.unsqueeze(1),
            labels.unsqueeze(0),
            False
        )
