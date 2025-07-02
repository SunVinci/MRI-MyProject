import torch
import torch.nn as nn


def complex_mul(a, b):
    return torch.complex(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    )


def complex_conj(x):
    return torch.complex(x.real, -x.imag)


def complex_scale(x, scale):
    if isinstance(x, (tuple, list)) and len(x) == 2:
        x = torch.complex(x[0], x[1])

    if isinstance(scale, torch.Tensor) and not torch.is_floating_point(scale):
        scale = scale.float()
    return torch.complex(x.real * scale, x.imag * scale)


class Smaps(nn.Module):
    def forward(self, image, smaps):

        #print("Smaps forward image shape before and dtype",image.shape,image.dtype)
        #print("Smaps forward smaps shape",smaps.shape)
        image = to_complex(image)
        if image.ndim == 4:
            image = image.unsqueeze(-1)
        #print("Smaps forward image shape after", image.shape)
        smaps = to_complex(smaps)

        #print("Smaps forward smaps shape after and dtype", smaps.shape,smaps.dtype)
        return complex_mul(image, smaps)


class SmapsAdj(nn.Module):
    def forward(self, coilimg, smaps):
        return torch.sum(complex_mul(coilimg, complex_conj(smaps)), dim=-1)


class MaskKspace(nn.Module):
    def forward(self, kspace, mask):
        return complex_scale(kspace, mask)


class FFT2(nn.Module):
    def forward(self, x):
        return torch.fft.fft2(x, norm='ortho')

class FFT2c(nn.Module):
    def forward(self, x):
        x = torch.fft.fftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x, norm='ortho')
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        return x

class IFFT2(nn.Module):
    def forward(self, x):
        return torch.fft.ifft2(x, norm='ortho')

class IFFT2c(nn.Module):
    def forward(self, x):
        x = torch.fft.fftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x, norm='ortho')
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        return x


class ForwardOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft2 = FFT2()
        self.mask = MaskKspace()

    def forward(self, image, mask):
        kspace = self.fft2(image)
        return self.mask(kspace, mask)


class AdjointOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.ifft2 = IFFT2()
        self.mask = MaskKspace()

    def forward(self, kspace, mask):
        masked = self.mask(kspace, mask)
        return self.ifft2(masked).unsqueeze(-1)


class MulticoilForwardOp(nn.Module):
    def __init__(self, center=False):
        super().__init__()
        self.fft2 = FFT2c() if center else FFT2()
        self.masker = MaskKspace()
        self.smapper = Smaps()

    def forward(self, image, mask, smaps):
        print(f"Image shape: {image.shape}")
        print(f"Smaps shape: {smaps.shape}")

        coilimg = self.smapper(image, smaps)
        kspace = self.fft2(coilimg)
        return self.masker(kspace, mask)


class MulticoilAdjointOp(nn.Module):
    def __init__(self, center=False):
        super().__init__()
        self.ifft2 = IFFT2c() if center else IFFT2()
        self.masker = MaskKspace()
        self.adj_smapper = SmapsAdj()

    def forward(self, kspace, mask, smaps):
        masked = self.masker(kspace, mask)
        coilimg = self.ifft2(masked)
        img = self.adj_smapper(coilimg, smaps)
        return img.unsqueeze(-1)



def ensure_complex_image(x):

    if x.dtype in (torch.complex64, torch.complex128):
        x = torch.view_as_real(x)
        if x.shape[-2] == 1:  # 特别情况：view_as_real 结果是 [..., 1, 2]
            x = x.squeeze(-2)  # 去掉多余的那一维
        print("🔁 view_as_real →", x.shape)
        return x

    if x.ndim == 5 and x.shape[-1] == 1:
        x = x.squeeze(-1)  # (B, T, H, W)
        print("🔁 squeezed →", x.shape)

    if x.ndim == 4:
        x = torch.stack([x, torch.zeros_like(x)], dim=-1)  # (B, T, H, W, 2)
        print("🔁 stacked →", x.shape)
        return x

    if x.ndim == 5 and x.shape[-1] == 2:
        print("✅ already (B,T,H,W,2) →", x.shape)
        return x

    raise ValueError(f"Invalid shape in ensure_complex_image: {x.shape}")


def center_crop_to_match(source, target):

    assert source.dim() == target.dim(), "输入张量维度不匹配"
    assert source.shape[0] == target.shape[0], "批次维度必须相同"
    assert source.shape[1] == target.shape[1], "通道维度必须相同"

    slices = []

    for i in range(2, source.dim()):
        if source.shape[i] == target.shape[i]:
            slices.append(slice(None))  # 无需裁剪
        else:
            diff = source.shape[i] - target.shape[i]
            assert diff >= 0, f"维度 {i} 无法裁剪：source 比 target 小"
            start = diff // 2
            slices.append(slice(start, start + target.shape[i]))

    final_slices = [slice(None), slice(None)] + slices
    return source[tuple(final_slices)]

def to_complex(x):
    """
    输入：x 是 float[..., 2] 或 complex
    返回：complex tensor，shape [...], dtype complex
    """
    if torch.is_complex(x):
        return x
    elif x.dtype in (torch.float32, torch.float64) and x.shape[-1] == 2:
        return torch.view_as_complex(x.contiguous())
    else:
        raise TypeError(f"Unsupported type for complex conversion: {x.dtype}, shape: {x.shape}")

