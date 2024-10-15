import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as edt

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']

class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha
        self.min_loss = float('inf')
        self.max_loss = float('-inf')

    @torch.no_grad()
    def update_min_max_loss(self, loss):
        self.min_loss = min(self.min_loss, loss)
        self.max_loss = max(self.max_loss, loss)

    @torch.no_grad()
    def normalize_loss(self, loss):
        return (loss - self.min_loss) / (self.max_loss - self.min_loss + 1e-8)

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(self, pred: torch.Tensor, target: torch.Tensor, debug=False) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y) or (b, 1, x, y, z)
        target: (b, 1, x, y) or (b, 1, x, y, z)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert target.dim() == pred.dim(), "Target should have the same dimension as prediction"

        device = pred.device  # Get the device of the input tensor

        # Apply threshold to pred and target
        pred_binary = (pred >= 0.5).float()  # Convert values >= 0.5 to 1, and < 0.5 to 0
        target_binary = (target >= 0.5).float()  # Convert values >= 0.5 to 1, and < 0.5 to 0

        # Compute distance transforms
        pred_dt = torch.from_numpy(self.distance_field(pred_binary.cpu().numpy())).float().to(device)  # Move to the same device
        target_dt = torch.from_numpy(self.distance_field(target_binary.cpu().numpy())).float().to(device)  # Move to the same device

        # Compute Hausdorff distance loss
        pred_error = (pred_binary - target_binary) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        class_loss = dt_field.mean()

        self.update_min_max_loss(class_loss.item())
        normalized_loss = self.normalize_loss(class_loss)

        if debug:
            return (
                normalized_loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )
        else:
            return normalized_loss



class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.hd_loss = HausdorffDTLoss(alpha=2.0)

    def forward(self, input, target):

        hd_loss_value = self.hd_loss(input, target)

        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num



        # Combine the losses
        return 0.5 * bce + dice + 0.5 * hd_loss_value


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
