import torch
import torch.nn as nn
from model.utils import MulticoilForwardOp, MulticoilAdjointOp, to_complex


class Scalar(nn.Module):
    def __init__(self, init=1.0, train_scale=1.0):
        super().__init__()
        self.init = init
        self.train_scale = train_scale
        self.weight_param = nn.Parameter(torch.tensor([init], dtype=torch.float32))

    @property
    def weight(self):
        return torch.clamp(self.weight_param, min=0.0) * self.train_scale

    def forward(self, x):
        return x * self.weight.view(*([1] * x.ndim))  # 广播支持任意维度


class InfoShareLayer(nn.Module):
    def __init__(self, center=True):
        super().__init__()
        self.tau_ksp = Scalar(init=0.5)
        self.tau_img = Scalar(init=0.5)
        self.forward_op = MulticoilForwardOp(center=center)
        self.adjoint_op = MulticoilAdjointOp(center=center)

    def forward(self, image, kspace, mask, smaps):
        """
        Args:
            image:  [B, T, H, W, 2] float 模拟复数
            kspace: [B, T, H, W, C] complex64
            mask:   [B, T, 1, W, 1] bool
            smaps:  [B, 1, H, W, C] complex64
        Returns:
            new_image: [B, T, H, W, 1] complex64
            new_kspace: [B, T, H, W, C] complex64
        """

        image = to_complex(image)  # [B, T, H, W]


        kspace_from_image = self.forward_op(image, mask, smaps)  # [B, T, H, W, C]
        tau_ksp_weight = self.tau_ksp.weight.view(1, 1, 1, 1, 1)
        new_kspace = self.tau_ksp(kspace) + (1.0 - tau_ksp_weight) * kspace_from_image  # 保持 [B,T,H,W,C]


        image_from_kspace = self.adjoint_op(kspace, mask, smaps)  # [B, T, H, W]
        tau_img_weight = self.tau_img.weight.view(1, 1, 1, 1, 1)

        new_image = self.tau_img(image) + (1.0 - tau_img_weight.squeeze(-1)) * image_from_kspace.squeeze(-1)  # [B, T, H, W]


        new_image = new_image.unsqueeze(-1)  # [B, T, H, W, 1]

        return new_image, new_kspace
