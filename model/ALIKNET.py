import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ksp import KspNetAttention
from model.info_share_layer import InfoShareLayer, Scalar
from model.image_lowrank_net import ComplexAttentionLSNet
from model.utils import MulticoilForwardOp, MulticoilAdjointOp, complex_scale, ensure_complex_image

import gc
import tracemalloc
import psutil, os

def log_memory(tag=""):
    process = psutil.Process(os.getpid())
    ram = process.memory_info().rss / 1024**2
    print(f"[MEM] {tag} | RAM: {ram:.2f} MB | GC objects: {len(gc.get_objects())}")


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
        self.KspNet = nn.ModuleList([KspNetAttention(in_ch=10,out_ch=10) for _ in range(num_iter)])
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
        sub_y, mask, smaps = constants
        y_real = y.real
        y_imag = y.imag

        #log_memory(f"Start update_xy[{i}]")
        y = self.KspNet[i](y_real, y_imag)
        #log_memory(f"After  KspNet[{i}]")

        x = ensure_complex_image(x)
        #log_memory(f"Before ImgLrNet[{i}]")
        x = self.ImgLrNet[i](x, self.S_end)
        #log_memory(f"After  ImgLrNet[{i}]")
        #log_memory(f"Before Ksp DC[{i}]")
        y = self.ksp_dc(y, mask, sub_y)
        #log_memory(f"After  Ksp DC[{i}]")

        #log_memory(f"Before ImgDC[{i}]")
        x = self.ImgDC[i](x, sub_y, mask, smaps)
        #log_memory(f"After  ImgDC[{i}]")

        #log_memory(f"Before ISL[{i}]")
        x, y = self.ISL[i](x, y, mask, smaps)
        #log_memory(f"After  ISL[{i}]")

        #log_memory(f"End update_xy[{i}]")

        return x, y

    def forward(self, x, y, mask, smaps):
        constants = (y, mask, smaps)
        for i in range(self.S_end):
            x, y = self.update_xy(x, y, i, constants)
        return y, x