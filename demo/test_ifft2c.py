import torch
import numpy as np

# 设置 GPU 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 模拟一个复杂 k-space 数据：shape [B, T, C, H, W]
B, T, C, H, W = 2, 3, 8, 128, 128  # 可以根据你实际修改
real = torch.randn(B, T, C, H, W)
imag = torch.randn(B, T, C, H, W)
kspace_tensor = torch.complex(real, imag).to(torch.complex64).to(device)

# 中心化 IFFT2 函数
def ifft2c_centered(x):
    x = torch.fft.fftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x

# 测试 ifft2c 是否正常运行
try:
    image_tensor = ifft2c_centered(kspace_tensor)
    print("ifft2c finished successfully.")
    print("Output shape:", image_tensor.shape)
    print("NaN:", torch.isnan(image_tensor).any().item())
    print("Inf:", torch.isinf(image_tensor).any().item())
except RuntimeError as e:
    print("cuFFT RuntimeError:", str(e))
