import torch
import torch.nn.functional as F


class DistillationLoss(torch.nn.Module):

    def __init__(self, alpha: float = 0.5, loss_type: str = "hard"):
        super(DistillationLoss, self).__init__()
        self._loss_type = loss_type
        self._base_loss_fn = torch.nn.CrossEntropyLoss
        self.alpha = alpha

    def forward(self, classification_output, classification_labels, distillation_output, distillation_labels):
        base_loss = self._base_loss_fn(classification_output, classification_labels)

        if self._loss_type == "soft":
            pass
        elif self._loss_type == "hard":
            pass
        else:
            raise ValueError("Distillation type not supported: {}".format(self._loss_type))
