import torch
import torch.nn as nn
import math
from copy import deepcopy
import logging
import torch.distributed as dist 

from ..base_collector import BaseCollector
from ...misc import utils
from ...models import BatchNormMLP

# modified from SimSiamCollector!

class SimSiamFullCollector(BaseCollector):
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
        layer_size_list=[2048, 512, 2048],
        relu_list=[True, False],
        bn_list=[True, False],
        is_same_source=True,
        first_bn=False,
        *args,
        **kwargs
    ):
        super(SimSiamFullCollector, self).__init__(*args, **kwargs) 
        self.is_same_source = is_same_source
        self.predictor = BatchNormMLP(
            layer_size_list=layer_size_list,
            relu_list=relu_list,
            bn_list=bn_list
        )
        self.bn = nn.BatchNorm1d(layer_size_list[0]) if first_bn else nn.Identity()
    
    def forward(self, data, embeddings, labels) -> tuple:
        """
        For simplicity, two data streams will be combined together and be passed through ``embeddings`` parameter. In function ``collect``, two data streams will be split (first half for first stream; second half for second stream).

        Args:
            data (torch.Tensor):
                A batch of key images (**not used**). size: :math:`B \\times C \\times H \\times W`
            embeddings (torch.Tensor):
                A batch of query embeddings. size: :math:`2B \\times dim`
            labels (torch.Tensor):
                Labels of the input. size: :math:`2B \\times 1`
        """
        # split two streams
        # compute p
        embeddings = self.bn(embeddings)
        p1 = self.predictor(embeddings)

        metric_mat = 1.0 * (
            self.metric(p1, embeddings.detach())
        )
        return (
            metric_mat,
            labels.unsqueeze(1),
            labels.unsqueeze(0),
            self.is_same_source
        )
