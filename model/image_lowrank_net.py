import torch
import torch.nn as nn
import torch.nn.functional as F
from model.image_net import ComplexUNet2Dt
from model.low_rank_net import LNetXYTBatch


def complex_scale(x, scale):
    # x: complex tensor, scale: float scalar
    return torch.complex(x.real * scale, x.imag * scale)


class Scalar(nn.Module):
    def __init__(self, init=1.0, train_scale=1.0):
        super().__init__()
        self.scale = train_scale
        self.weight = nn.Parameter(torch.tensor([init], dtype=torch.float32))

    def forward(self, x):
        return complex_scale(x, self.weight * self.scale)


class ComplexAttentionLSNet(nn.Module):
    def __init__(self, num_patches=80, image_size=(192, 156), patch_size=(32, 32), time_size=25):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 初始化网络组件
        self.R = ComplexUNet2Dt(in_channels=1, out_channels=1, base_filters=12, time_size=time_size)
        self.D = LNetXYTBatch(num_patches=num_patches)
        self.tau = Scalar(init=0.1)
        self.p_weight = Scalar(init=0.5)

        self.current_patch_info = None

    def extract_patches(self, x):
        B, T, H, W = x.shape

        # 动态计算网格尺寸（考虑填充）
        grid_x = (H + self.patch_size[0] - 1) // self.patch_size[0]
        grid_y = (W + self.patch_size[1] - 1) // self.patch_size[1]

        # 计算填充量
        pad_h = grid_x * self.patch_size[0] - H
        pad_w = grid_y * self.patch_size[1] - W

        # 应用填充
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))

        # 提取补丁
        x = x_padded.view(B * T, 1, x_padded.size(-2), x_padded.size(-1))
        patches = x.unfold(2, self.patch_size[0], self.patch_size[0])
        patches = patches.unfold(3, self.patch_size[1], self.patch_size[1])

        # 计算实际补丁数量
        actual_num_patches = patches.size(2) * patches.size(3)
        patches = patches.contiguous().view(B, T, actual_num_patches, self.patch_size[0], self.patch_size[1])

        # 保存补丁信息
        self.current_patch_info = {
            'grid_x': grid_x,
            'grid_y': grid_y,
            'pad_h': pad_h,
            'pad_w': pad_w,
            'actual_num_patches': actual_num_patches
        }

        #print(f"提取补丁: grid_x={grid_x}, grid_y={grid_y}, 补丁数={actual_num_patches}")
        return patches

    def recover_patches(self, x, num_patches):
        B, T, N, Px, Py = x.shape

        # 提取补丁信息
        grid_x = self.current_patch_info['grid_x']
        grid_y = self.current_patch_info['grid_y']

        # 验证补丁数量
        assert N == grid_x * grid_y, f"补丁数量不匹配: {N} vs {grid_x * grid_y}"

        # 重塑补丁
        x = x.view(B, T, grid_x, grid_y, Px, Py)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, T, grid_x * Px, grid_y * Py)

        # 移除填充
        H_original, W_original = self.image_size
        x = x[..., :H_original, :W_original]

        #print(f"恢复图像: 尺寸={x.shape[-2:]}")
        return x

    def forward(self, x, num_iter=5):
        # x: (B, T, H, W, 2)
        x_complex = torch.complex(x[..., 0], x[..., 1])  # -> (B, T, H, W)

        # Denoising
        x_input = x_complex.unsqueeze(1)  # (B, 1, T, H, W)
        den = self.R(x_input).squeeze(1)  # (B, T, H, W)
        p = x_complex - self.tau(den) / num_iter

        # Low-rank operation on patches
        patches = self.extract_patches(x_complex)  # (B, T, N, Px, Py)
        patches = patches.permute(0, 2, 1, 3, 4)  # (B, N, T, Px, Py)
        B, N, T, Px, Py = patches.shape
        patches_reshape = patches.reshape(B * N, T, Px, Py)  # (BN, T, Px, Py)

        # 批次分割处理大批次
        max_batch = 100
        if B * N > max_batch:
            q_patches_list = []
            for i in range(0, B * N, max_batch):
                batch_patches = patches_reshape[i:i + max_batch]
                q_batch = self.D(batch_patches)
                q_patches_list.append(q_batch)
            q_patches = torch.cat(q_patches_list, dim=0)
        else:
            q_patches = self.D(patches_reshape)

        q_patches = q_patches.view(B, N, T, Px, Py).permute(0, 2, 1, 3, 4)  # (B, T, N, Px, Py)
        q = self.recover_patches(q_patches, N)  # (B, T, H, W)

        # Weighted combination
        weighted_p = self.p_weight(p)
        q_weight = 1.0 - self.p_weight.weight.item()
        weighted_q = complex_scale(q, q_weight)

        # 确保维度匹配

        x_out = weighted_p + weighted_q  # (B, T, H, W)
        x_out = torch.stack([x_out.real, x_out.imag], dim=-1)  # (B, T, H, W, 2)
        return x_out