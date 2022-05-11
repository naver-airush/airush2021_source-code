import torch
import torch.nn.functional as F


class LabelSmoothingNLLLoss(torch.nn.Module):
    """Pytorch implementation of label smoothed NLL loss retrieved from
    https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/166833#930136
    """

    def __init__(self, smoothing: float = 0.1, reduction="mean", weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, log_preds, target, ignore_index=-100):
        """
        log_preds: [bs, V, T]
        target: [bs, T]
        """
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(log_preds.device)

        mask = (target != ignore_index).float().unsqueeze(1)  # [bs, 1, T]
        masked_log_preds = log_preds * mask

        loss = self.reduce_loss(-masked_log_preds.mean(dim=1))
        nll = F.nll_loss(log_preds,
                         target,
                         reduction=self.reduction,
                         weight=self.weight,
                         ignore_index=ignore_index)
        return self.linear_combination(loss, nll)
