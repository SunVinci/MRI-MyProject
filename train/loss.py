import pytorch_msssim
import torch.nn as nn

import torch
import torch.nn as nn
import pytorch_msssim

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim_fn = pytorch_msssim.SSIM(data_range=1.0, size_average=True, channel=1)
        self.alpha = alpha

    def forward(self, pred, target):
        # pred, target: [B, T, H, W]
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)

        B, T, H, W = pred.shape
        l1_loss = self.l1(pred, target)

        ssim_total = 0.0
        for t in range(T):
            pred_frame = pred[:, t:t+1, :, :]  # shape: [B,1,H,W]
            target_frame = target[:, t:t+1, :, :]
            ssim_total += 1 - self.ssim_fn(pred_frame, target_frame)

        ssim_loss = ssim_total / T
        return self.alpha * ssim_loss + (1 - self.alpha) * l1_loss

