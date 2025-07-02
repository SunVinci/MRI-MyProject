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
        self.pool = nn.AdaptiveMaxPool3d((time_size, 1, 1))
        self.fc1 = nn.Linear(time_size*2, bottle_size * 2)
        self.fc2 = nn.Linear(bottle_size * 2, time_size * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, T, H, W] complex tensor
        real = x.real.mean(dim=(-2, -1))  # [B, C, T]
        imag = x.imag.mean(dim=(-2, -1))  # [B, C, T]

        x_cat = torch.cat([real, imag], dim=-1)  # [B, C, 2*T]
        B, C, _ = x_cat.shape
        x_cat = x_cat.view(B * C, -1)  # [B*C, 2*T]

        x_exc = self.fc2(self.fc1(x_cat))  # [B*C, 2*T]
        x_exc = self.sigmoid(x_exc).view(B, C, self.time_size * 2, 1, 1)  # [B, C, 2*T, 1, 1]

        out_real = x.real * x_exc[:, :, :self.time_size]
        out_imag = x.imag * x_exc[:, :, self.time_size:]
        return torch.complex(out_real, out_imag)


class ComplexUNet2Dt(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32, time_size=25):
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
        #print(f"Input shape: {x.shape}")  # 应该是[B, in_channels, T, H, W]

        e1 = self.enc1(x)
        #print(f"enc1 output shape: {e1.shape}")

        e1_real = e1.real
        e1_imag = e1.imag
        p1_real = self.pool1(e1_real)
        p1_imag = self.pool1(e1_imag)
        p1 = torch.complex(p1_real, p1_imag)
        #print(f"pool1 output shape: {p1.shape}")

        e2 = self.enc2(p1)
        #print(f"enc2 output shape: {e2.shape}")

        e2_real = e2.real
        e2_imag = e2.imag
        p2_real = self.pool2(e2_real)
        p2_imag = self.pool2(e2_imag)
        p2 = torch.complex(p2_real, p2_imag)
        #print(f"pool2 output shape: {p2.shape}")

        b = self.bottleneck(p2)
        #print(f"bottleneck output shape: {b.shape}")

        b_real = b.real
        b_imag = b.imag
        up2_real = self.up2(b_real)
        up2_imag = self.up2(b_imag)
        up2 = torch.complex(up2_real, up2_imag)
        #print(f"up2 (before crop) shape: {up2.shape}")

        up2 = self.channel_adjust(up2)
        #print(f"up2 (after adjust) shape: {up2.shape}")

        # 裁剪 up2 和 e2 以保证可以拼接
        up2 = center_crop_to_match(up2, e2)
        #print(f"up2 (after crop) shape: {up2.shape}")
        #print(f"e2 shape: {e2.shape}")

        cat2 = torch.cat([up2, e2], dim=1)
        #print(f"cat2 shape: {cat2.shape}")

        d2 = self.dec2(cat2)
        #print(f"dec2 output shape: {d2.shape}")
        d2 = self.attn1(d2)

        d2_real = d2.real
        d2_imag = d2.imag
        up1_real = F.interpolate(d2_real, size=e1.shape[2:], mode='trilinear', align_corners=False)
        up1_imag = F.interpolate(d2_imag, size=e1.shape[2:], mode='trilinear', align_corners=False)
        up1 = torch.complex(up1_real, up1_imag)
        #print(f"up1 (before crop) shape: {up1.shape}")

        up1 = self.channel_adjust_up1(up1)
        #print(f"up1 (after adjust) shape: {up1.shape}")

        # 裁剪 up1 和 e1 以保证可以拼接
        up1 = center_crop_to_match(up1, e1)
        #print(f"up1 (after crop) shape: {up1.shape}")
        #print(f"e1 shape: {e1.shape}")

        cat1 = torch.cat([up1, e1], dim=1)
        #print(f"cat1 shape: {cat1.shape}")

        d1 = self.dec1(cat1)
        #print(f"dec1 output shape: {d1.shape}")
        d1 = self.attn2(d1)

        out = self.out_conv(d1)
        #print(f"Output shape: {out.shape}")

        return out

