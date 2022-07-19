import numpy as np 
import torch

from .base_selector import BaseSelector
from ...config.setting.core_setting import (
    INDICES_TUPLE,
    INDICES_FLAG
)

class HardPairSelector(BaseSelector):
    """
    A self-defined selector according to distance weighted sampling method
    """
    def __init__(self, hardneg_cutoff=0.5, is_similarity=False, **kwargs):
        super(HardPairSelector, self).__init__(**kwargs)
        self.hardneg_cutoff = hardneg_cutoff
        self.is_similarity = is_similarity
    
    def forward(
        self,
        metric_mat,
        row_labels,
        col_labels,
        is_same_source=False
    ) -> tuple:
        bs = metric_mat.size(0)
        device = metric_mat.device
        dtype = metric_mat.dtype
        # pos and neg mask
        matches = (row_labels == col_labels).byte()
        diffs = matches ^ 1

        if not self.is_similarity:
            hardneg_cutoff = min(
                torch.max(metric_mat[matches.bool()]).item(), self.hardneg_cutoff
            )
            metric_mat_to_weights = metric_mat.clamp(min=hardneg_cutoff).detach()
        else:
            hardneg_cutoff = max(
                torch.min(metric_mat[matches.bool()]).item(), self.hardneg_cutoff
            )
            metric_mat_to_weights = metric_mat.clamp(max=hardneg_cutoff).detach()
        

        # construct the tuple
        if is_same_source:
            matches.fill_diagonal_(0)
        
        has_pos_mask = torch.where(
            torch.sum(matches, dim=-1) > 0
        )[0]

        ## anchor indices
        a_ids = torch.arange(bs).to(device)
        ## positive indices
        p_ids = torch.multinomial(
            input=matches[has_pos_mask].float(),
            num_samples=1,
            replacement=True
        ).flatten()
        pos_pairs = torch.stack([a_ids[has_pos_mask], p_ids], dim=1)
        pos_flags = torch.ones(pos_pairs.shape[0], 1)

        ## negative indices
        if not self.is_similarity:
            metric_mat_to_weights[~diffs.bool()] = torch.finfo(dtype).max
            n_ids = torch.argmin(
                metric_mat_to_weights, dim=-1
            )
        else:
            metric_mat_to_weights[~diffs.bool()] = torch.finfo(dtype).min
            n_ids = torch.argmax(
                metric_mat_to_weights, dim=-1
            )
        neg_pairs = torch.stack([a_ids, n_ids], dim=1)
        neg_flags = torch.zeros(neg_pairs.shape[0], 1)
        tuples = torch.cat([pos_pairs, neg_pairs], dim=0)
        flags = torch.cat([pos_flags, neg_flags], dim=0).byte()

        indices_tuple = {
            INDICES_TUPLE: tuples,
            INDICES_FLAG: flags
        }
        weight = None
        return (
            metric_mat,
            row_labels,
            col_labels,
            is_same_source,
            indices_tuple,
            weight,
        )
