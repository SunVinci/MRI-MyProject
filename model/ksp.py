import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexSECoilLayer(nn.Module):
    def __init__(self, coil_size, bottle_size=2):
        super().__init__()
        self.coil_size = coil_size
        self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc1 = nn.Linear(coil_size * 2, bottle_size * 2)
        self.fc2 = nn.Linear(bottle_size * 2, coil_size * 2)
        self.activation = nn.Sigmoid()

    def forward(self, x_real, x_imag):
        x = torch.cat([x_real, x_imag], dim=1)
        x_pooled = self.pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        x_se = self.fc1(x_pooled)
        x_se = self.fc2(x_se)
        x_se = self.activation(x_se)

        scale_real = x_se[:, :self.coil_size].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        scale_imag = x_se[:, self.coil_size:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return x_real * scale_real, x_imag * scale_imag

class KspNetAttention(nn.Module):
    def __init__(self, in_ch=15, out_ch=25):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch * 2, 64, kernel_size=3, padding=1)
        self.se1 = ComplexSECoilLayer(coil_size=32)

        self.conv2 = nn.Conv3d(64, 16, kernel_size=3, padding=1)
        self.se2 = ComplexSECoilLayer(coil_size=8)

        self.conv3 = nn.Conv3d(16, out_ch * 2, kernel_size=3, padding=1)

    def forward(self, x_real, x_imag):

        # [B, T, H, W, C] -> [B, C, T, H, W]
        x_real = x_real.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        x_imag = x_imag.permute(0, 4, 1, 2, 3)

        x = torch.cat([x_real, x_imag], dim=1)  # (B, 2C, T, H, W)

        x = self.conv1(x)
        x_real, x_imag = torch.chunk(x, 2, dim=1)
        x_real, x_imag = self.se1(x_real, x_imag)

        x = torch.cat([x_real, x_imag], dim=1)
        x = self.conv2(x)
        x_real, x_imag = torch.chunk(x, 2, dim=1)
        x_real, x_imag = self.se2(x_real, x_imag)

        x = torch.cat([x_real, x_imag], dim=1)
        x = self.conv3(x)
        x_real, x_imag = torch.chunk(x, 2, dim=1)

        # Optional: Convert back to [B, T, H, W, C] if needed later
        x_real = x_real.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        x_imag = x_imag.permute(0, 2, 3, 4, 1)

        return x_real, x_imag