import torch

from ...misc import loss_function as l_f 
from ..base_loss import BaseLoss

class PosPairLoss(BaseLoss):
    """
    Designed for SimSiam.

    paper: `Exploring Simple Siamese Representation Learning <https://arxiv.org/abs/2011.10566>`_
    """
    def __init__(
        self,
        **kwargs
    ):
        super(PosPairLoss, self).__init__(**kwargs)
        self.to_record_list = [
            "pos_cosine_mean",
            "neg_cosine_mean"
        ]
    
    def required_metric(self):
        return ["cosine"]
    
    def compute_loss(
        self,
        metric_mat,
        row_labels,
        col_labels,
        indices_tuple,
        weights=None,
        is_same_source=False,
    ) -> torch.Tensor:
        pos_mask = (row_labels == col_labels)
        neg_mask = ~pos_mask
        neg_pair = metric_mat[neg_mask]
        
        if is_same_source:
            pos_mask.fill_diagonal_(False)
        pos_pair = metric_mat[pos_mask]
        #loss = torch.mean(torch.exp(1-pos_pair)*(-pos_pair))
        loss = torch.mean(-pos_pair)

        # stat
        self.pos_cosine_mean = torch.mean(pos_pair)
        self.neg_cosine_mean = torch.mean(neg_pair)
        return loss
    
