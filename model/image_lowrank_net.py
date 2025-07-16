import torch
import torch.nn as nn
import torch.nn.functional as F
from model.image_net import ComplexUNet2Dt
from model.low_rank_net import LNetXYTBatch

def complex_scale(x, scale):
    return torch.complex(x.real * scale, x.imag * scale)

class Scalar(nn.Module):
    def __init__(self, init=1.0, train_scale=1.0):
        super().__init__()
        self.scale = train_scale
        self.weight = nn.Parameter(torch.tensor([init], dtype=torch.float32))

    def forward(self, x):
        return complex_scale(x, self.weight * self.scale)

class ComplexAttentionLSNet(nn.Module):
    def __init__(self, T=5, patch_size=(32, 32)):
        super().__init__()
        self.tau = Scalar(init=0.1)
        self.p_weight = Scalar(init=0.5)
        self.patch_size = patch_size

        # 静态初始化 R 和 D
        self.R = ComplexUNet2Dt(in_channels=1, out_channels=1, base_filters=12, time_size=T)
        self.D = LNetXYTBatch(patch_size=patch_size)

        # 存储补丁信息用于 recover
        self.current_patch_info = None

    def extract_patches(self, x):
        B, T, H, W = x.shape
        Px, Py = self.patch_size

        pad_h = (Px - H % Px) % Px
        pad_w = (Py - W % Py) % Py

        x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        H_pad, W_pad = x_padded.shape[-2:]

        self.current_patch_info = {
            'H': H, 'W': W,
            'H_pad': H_pad, 'W_pad': W_pad,
            'pad_h': pad_h, 'pad_w': pad_w,
            'Px': Px, 'Py': Py,
        }

        x_reshaped = x_padded.view(B * T, 1, H_pad, W_pad)
        patches = x_reshaped.unfold(2, Px, Px).unfold(3, Py, Py)
        patches = patches.contiguous().view(B, T, -1, Px, Py)
        return patches

    def recover_patches(self, x):
        B, T, N, Px, Py = x.shape
        info = self.current_patch_info
        num_patches_x = info['H_pad'] // Px
        num_patches_y = info['W_pad'] // Py
        assert N == num_patches_x * num_patches_y

        x = x.view(B, T, num_patches_x, num_patches_y, Px, Py)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, T, num_patches_x * Px, num_patches_y * Py)
        return x[:, :, :info['H'], :info['W']]

    def forward(self, x, num_iter=5):
        B, T, H, W, _ = x.shape
        x_complex = torch.complex(x[..., 0], x[..., 1])  # (B, T, H, W)

        # 图像正则化（UNet）
        x_input = x_complex.unsqueeze(1)  # (B, 1, T, H, W)
        den = self.R(x_input).squeeze(1)
        p = x_complex - self.tau(den) / num_iter

        # Patch -> D模块
        patches = self.extract_patches(x_complex)  # (B, T, N, Px, Py)
        patches = patches.permute(0, 2, 1, 3, 4)
        B_, N, T_, Px, Py = patches.shape
        patches_reshape = patches.reshape(B_ * N, T_, Px, Py)

        # 分批处理（防爆内存）
        max_batch = 64
        if B_ * N > max_batch:
            q_patches_list = []
            for i in range(0, B_ * N, max_batch):
                q_batch = self.D(patches_reshape[i:i + max_batch])
                q_patches_list.append(q_batch)
            q_patches = torch.cat(q_patches_list, dim=0)
        else:
            q_patches = self.D(patches_reshape)

        q_patches = q_patches.view(B_, N, T_, Px, Py).permute(0, 2, 1, 3, 4)
        q = self.recover_patches(q_patches)

        # 融合图像
        weighted_p = self.p_weight(p)
        weighted_q = complex_scale(q, 1.0 - self.p_weight.weight.item())

        # 对齐裁剪
        Hp, Wp = weighted_p.shape[2:]
        weighted_q = weighted_q[:, :, :Hp, :Wp]

        x_out = weighted_p + weighted_q
        return torch.stack([x_out.real, x_out.imag], dim=-1)