import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import center_crop_to_match

class ComplexConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, activation='modrelu'):
        super().__init__()
        self.real_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.imag_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation

    def forward(self, x):
        real = self.real_conv(x.real) - self.imag_conv(x.imag)
        imag = self.real_conv(x.imag) + self.imag_conv(x.real)
        out = torch.complex(real, imag)

        if self.activation == 'modrelu':
            mag = torch.abs(out)
            phase = torch.angle(out)
            return mag * torch.exp(1j * phase)
        return out

class ComplexSETimeLayer(nn.Module):
    def __init__(self, time_size, bottle_size=2):
        super().__init__()
        self.time_size = time_size
        self.fc1 = nn.Linear(time_size * 2, bottle_size * 2)
        self.fc2 = nn.Linear(bottle_size * 2, time_size * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, T, H, W] complex
        real = x.real.mean(dim=(-2, -1))  # [B, C, T]
        imag = x.imag.mean(dim=(-2, -1))  # [B, C, T]
        x_cat = torch.cat([real, imag], dim=-1)  # [B, C, 2*T]
        B, C, _ = x_cat.shape
        x_cat = x_cat.view(B * C, -1)  # [B*C, 2*T]

        x_exc = self.fc2(self.fc1(x_cat))  # [B*C, 2*T]
        x_exc = self.sigmoid(x_exc).view(B, C, self.time_size * 2, 1, 1)

        scale_real = x_exc[:, :, :self.time_size]
        scale_imag = x_exc[:, :, self.time_size:]
        return torch.complex(x.real * scale_real, x.imag * scale_imag)

class ComplexUNet2Dt(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32, time_size=5):
        super().__init__()

        self.enc1 = nn.Sequential(
            ComplexConv3D(in_channels, base_filters, 3),
            ComplexConv3D(base_filters, base_filters, 3)
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            ComplexConv3D(base_filters, base_filters * 2, 3),
            ComplexConv3D(base_filters * 2, base_filters * 2, 3)
        )
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = nn.Sequential(
            ComplexConv3D(base_filters * 2, base_filters * 4, 3),
            ComplexConv3D(base_filters * 4, base_filters * 4, 3)
        )

        self.channel_adjust = ComplexConv3D(base_filters * 4, base_filters * 2, kernel_size=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.dec2 = nn.Sequential(
            ComplexConv3D(base_filters * 4, base_filters * 2, 3),
            ComplexConv3D(base_filters * 2, base_filters * 2, 3)
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.channel_adjust_up1 = ComplexConv3D(base_filters * 2, base_filters, kernel_size=1)

        self.dec1 = nn.Sequential(
            ComplexConv3D(base_filters * 2, base_filters, 3),
            ComplexConv3D(base_filters, base_filters, 3)
        )

        self.attn1 = ComplexSETimeLayer(time_size=time_size // 2)
        self.attn2 = ComplexSETimeLayer(time_size=time_size)
        self.out_conv = ComplexConv3D(base_filters, out_channels, kernel_size=1, padding=0, activation=None)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = torch.complex(self.pool1(e1.real), self.pool1(e1.imag))

        e2 = self.enc2(p1)
        p2 = torch.complex(self.pool2(e2.real), self.pool2(e2.imag))

        b = self.bottleneck(p2)
        up2 = torch.complex(self.up2(b.real), self.up2(b.imag))
        up2 = self.channel_adjust(up2)
        up2 = center_crop_to_match(up2, e2)

        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        d2 = self.attn1(d2)

        up1_real = F.interpolate(d2.real, size=e1.shape[2:], mode='trilinear', align_corners=False)
        up1_imag = F.interpolate(d2.imag, size=e1.shape[2:], mode='trilinear', align_corners=False)
        up1 = torch.complex(up1_real, up1_imag)
        up1 = self.channel_adjust_up1(up1)
        up1 = center_crop_to_match(up1, e1)

        d1 = self.dec1(torch.cat([up1, e1], dim=1))
        d1 = self.attn2(d1)

        out = self.out_conv(d1)
        return out
