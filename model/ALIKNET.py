import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ksp import KspNetAttention
from model.info_share_layer import InfoShareLayer, Scalar
from model.image_lowrank_net import ComplexAttentionLSNet
from model.utils import MulticoilForwardOp, MulticoilAdjointOp, complex_scale, ensure_complex_image


class DCGD(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = MulticoilForwardOp(center=True)
        self.AH = MulticoilAdjointOp(center=True)

    def forward(self, x, sub_ksp, mask, smaps):
        kspace_pred = self.A(x, mask, smaps)
        diff = complex_scale(sub_ksp - kspace_pred, mask)
        update = self.AH(diff, mask, smaps)


        update = update.squeeze(-1)  # [B, T, H, W]
        update_real = torch.view_as_real(update)  # [B, T, H, W, 2] float32

        return x + update_real


class A_LIKNet(nn.Module):
    def __init__(self, num_iter=8):
        super().__init__()

        self.S_end = num_iter
        self.ksp_dc_weight = Scalar(init=0.5)

        self.ISL = nn.ModuleList([InfoShareLayer(center=True) for _ in range(num_iter)])
        self.KspNet = nn.ModuleList([KspNetAttention(in_ch=15,out_ch=15) for _ in range(num_iter)])
        self.ImgLrNet = nn.ModuleList([ComplexAttentionLSNet() for _ in range(num_iter)])
        self.ImgDC = nn.ModuleList([DCGD() for _ in range(num_iter)])

    def ksp_dc(self, kspace, mask, sub_ksp):
        mask = mask.float()
        masked_pred_ksp = complex_scale(kspace, mask)
        scaled_pred_ksp = self.ksp_dc_weight(masked_pred_ksp)
        scaled_sampled_ksp = complex_scale(sub_ksp, 1.0 - self.ksp_dc_weight.weight.view(1, 1, 1, 1, 1))
        other_points = complex_scale(kspace, (1 - mask))


        return scaled_pred_ksp + scaled_sampled_ksp + other_points

    def update_xy(self, x, y, i, constants):

        print("????????????????? X", x.shape, x.dtype)
        print("????????????????? Y", y.shape, y.dtype)


        sub_y, mask, smaps = constants
        y_real = y.real
        y_imag = y.imag
        y = self.KspNet[i](y_real, y_imag)

        y = torch.complex(y_real, y_imag)

        x = ensure_complex_image(x)

        print("A_LIKNet ImLrNet x.shape",x.shape, x.dtype)

        x = self.ImgLrNet[i](x, self.S_end)

        y = self.ksp_dc(y, mask, sub_y)

        print("A_LIKNet Before DC x", x.shape, x.dtype)

        x = self.ImgDC[i](x, sub_y, mask, smaps)

        print("A_LIKNet Before ISL x", x.shape, x.dtype)
        print("A_LIKNet Before ISL y", y.shape, y.dtype)

        x, y = self.ISL[i](x, y, mask, smaps)


        print("!!!!!!!!!!!!!!!!! X", x.shape, x.dtype)
        print("!!!!!!!!!!!!!!!!! Y", y.shape, y.dtype)

        return x, y

    def forward(self, x, y, mask, smaps):
        constants = (y, mask, smaps)
        for i in range(self.S_end):
            x, y = self.update_xy(x, y, i, constants)
        return y, x