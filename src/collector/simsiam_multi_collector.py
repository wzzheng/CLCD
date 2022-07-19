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

class SimSiamMultiCollector(BaseCollector):
    """
    Paper: `Exploring Simple Siamese Representation Learning <https://arxiv.org/abs/2011.10566>`_

    This method use none of the following to learn meaningful representations:

    1. negative sample pairs;
    2. large batches;
    3. momentum encoders.

    And a stop-gradient operation plays an essential role in preventing collapsing.
    """
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

        self.num_levels = len(reduction_size) - 1
        self.reduction = nn.ModuleList([
            BatchNormLayer(reduction_size[i], reduction_size[i+1], True, True, False) for i in range(self.num_levels)
        ])
        self.recover = nn.ModuleList([
            BatchNormLayer(reduction_size[i+1], reduction_size[0], False, last_bn, True) for i in range(self.num_levels)
        ])
        # self.drop = nn.Dropout(p=0.1)
    
    def forward(self, data, embeddings, labels):

        # split two streams
        N = embeddings.size(0)
        world_size = dist.get_world_size()
        assert N % (2*world_size) == 0
        embeddings = torch.chunk(embeddings, 2*world_size, dim=0)
        z1 = [embeddings[2*i] for i in range(world_size)]
        z2 = [embeddings[2*i+1] for i in range(world_size)]
        z1 = torch.cat(z1, dim=0)
        z2 = torch.cat(z2, dim=0)
            
        # z1, z2 = embeddings[:N//2], embeddings[N//2:]
        
        # pass through batchnorm layer
        z1, z2 = self.bn(z1), self.bn(z2)
        # list of metric mats of different levels
        metric_mats = []
        reduced1, reduced2 = z1, z2
        for i in range(self.num_levels):
            reduced1 = self.reduction[i](reduced1)
            recover1 = self.recover[i](reduced1)

            reduced2 = self.reduction[i](reduced2)
            recover2 = self.recover[i](reduced2)

            metric_mat = 0.5 * (
                self.metric(recover1, z2) +
                self.metric(recover2, z1)
            )
            metric_mats.append(metric_mat)
        
        return (
            metric_mats,
            labels.transpose(0, 1).unsqueeze(-1),
            labels.transpose(0, 1).unsqueeze(1),
            False
        )
