import torch

from gedml.core.misc import loss_function as l_f
from gedml.core.losses import BaseLoss

class MultiPosPairLoss(BaseLoss):
    """
    Designed for SimSiam.

    paper: `Exploring Simple Siamese Representation Learning <https://arxiv.org/abs/2011.10566>`_
    """
    def __init__(
        self,
        mask_rules=[[True]],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mask_rules = mask_rules

        self.to_record_list = []
        for i in range(len(mask_rules)):
            self.to_record_list += ["pos_mean_{}".format(i), "neg_mean_{}".format(i), "loss_{}".format(i)]
    
    def required_metric(self):
        return ["cosine"]
    
    def generate_mask(self, rules, row_labels, col_labels, is_same_source, device):
        num_samples = row_labels[0].shape[0]
        pos_mask = torch.ones(num_samples, num_samples, dtype=torch.bool, device=device)
        neg_mask = torch.ones(num_samples, num_samples, dtype=torch.bool, device=device)

        for i, rule in enumerate(rules):
            if rule == None:
                continue
            row_label = row_labels[i]
            col_label = col_labels[i]

            equal_mask = row_label == col_label
            neg_mask = neg_mask & (~equal_mask)
            if is_same_source and rule:
                equal_mask.fill_diagonal_(False)
            if rule:
                pos_mask = pos_mask & equal_mask
            else:
                pos_mask = pos_mask & (~equal_mask) 
        
        return pos_mask, neg_mask
    
    def compute_loss(
        self,
        metric_mat,
        row_labels,
        col_labels,
        indices_tuple,
        weights=None,
        is_same_source=False,
    ) -> torch.Tensor:
        device = metric_mat[0].device
        # num_samples = metric_mat[0].shape[0]
        # num_label_levels = row_labels.shape[0] // num_samples
        # row_labels = torch.chunk(row_labels, num_label_levels, dim=0)
        # col_labels = torch.chunk(col_labels, num_label_levels, dim=1)

        num_levels = len(metric_mat)

        total_loss = 0
        for i, mat in enumerate(metric_mat):
            pos_mask, neg_mask = self.generate_mask(
                self.mask_rules[i], row_labels, col_labels, is_same_source, device)
            
            pos_pair, neg_pair = mat[pos_mask], mat[neg_mask]

            #loss = torch.mean(torch.exp(1-pos_pair)*(-pos_pair))
            if i == num_levels - 1:
                neg_loss = torch.mean(neg_pair)
            else:
                neg_loss = 0

            loss = torch.mean(-pos_pair) + neg_loss
            total_loss += loss
            # stat
            self.__setattr__("loss_{}".format(i), loss)
            self.__setattr__("pos_mean_{}".format(i), torch.mean(pos_pair))
            self.__setattr__("neg_mean_{}".format(i), torch.mean(neg_pair))

        total_loss = total_loss / len(self.mask_rules)

        return total_loss
    
