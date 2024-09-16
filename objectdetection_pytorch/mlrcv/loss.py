import torch
import numpy as np
from typing import Optional

def focal_loss(pred: torch.Tensor, gt: torch.Tensor, alpha: Optional[int] = 2, beta: Optional[int] = 4) -> torch.Tensor:
    """
    This function computes the focal loss as described in the CenterNet paper:

    Args:
        - pred (torch.Tensor): predicted network output heatmap
        - gt (torch.Tensor): ground truth heatmap
        - alpha (int): alpha parameter of the focal loss (use the default value)
        - beta (int): beta parameter of the focal loss (use the default value)

    Returns:
        - loss (torch.Tensor): computed focal loss, a torch.Tensor with shape (1) (only one value)
    """
    if(gt == 1):
        focal_loss = -1 * torch.pow((1 - pred),alpha) * torch.log(pred)
        else:
            focal_loss = -1 * torch.pow((1 - pred),beta) * torch.pow(pred,alpha) * torch.log(1-pred)
            else:
                for i in range(gt.shape[0]):
                    for j in range(gt.shape[1]):
                        for k in range(gt.shape[2]):
                            for l in range(gt.shape[3]):
                                if(gt[i,j,k,l] == 1):
                                    focal_loss = -1 * torch.pow((1 - pred[i,j,k,l]),alpha) * torch.log(pred[i,j,k,l])
                                else:
                                    focal_loss = -1 * torch.pow((1 - pred[i,j,k,l]),beta) * torch.pow(pred[i,j,k,l],alpha) * torch.log(1-pred[i,j,k,l])
    
    loss = focal_loss
    # loss = None

    return loss

def smooth_l1_loss(pred: torch.Tensor, gt: torch.Tensor, sz_mask: np.ndarray) -> torch.Tensor:
    """
    This function computes the focal loss as described in the CenterNet paper:

    Args:
        - pred (torch.Tensor): predicted network output sizemap
        - gt (torch.Tensor): ground truth sizemap
        - sz_mask (numpy.ndarray): mask with the index to compute the loss (only the objects centers)

    Returns:
        - loss (torch.Tensor): computed smooth l1 loss, a torch.Tensor with shape (1) (only one value)
    """

    x = pred[sz_mask] - gt[sz_mask]
    smooth_l1 = torch.zeros_like(x)

    abs_x = torch.abs(x)
    smooth_l1 += torch.where(abs_x < 1, 0.5 * x ** 2, abs_x - 0.5)

    loss = smooth_l1.mean()
    
    return loss

def centerloss(pred_ht, pred_sz, gt_ht, gt_sz, sz_mask, loss_weights):
    # Binary mask loss
    ht_loss = focal_loss(pred_ht, gt_ht)

    # Regression L1 loss
    pred_sz = pred_sz.permute(0,2,3,1)
    gt_sz = gt_sz.permute(0,2,3,1)
    sz_loss = smooth_l1_loss(pred_sz, gt_sz, sz_mask)

    # Sum
    loss = loss_weights[0] * ht_loss + loss_weights[1] * sz_loss

    return loss, ht_loss , sz_loss
