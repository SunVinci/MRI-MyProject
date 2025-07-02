import torch
import torch.nn as nn
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from ALIKNET import A_LIKNet

def create_random_complex_tensor(shape, device='cpu'):
    real = torch.randn(shape, device=device)
    imag = torch.randn(shape, device=device)
    return torch.complex(real, imag)


def create_inputs(batch_size=1, nt=25, nx=192, ny=156, nc=15, device='cpu'):
    masked_img = create_random_complex_tensor((batch_size, nt, nx, ny, 1), device=device)     # (B, T, H, W, 1)
    masked_kspace = create_random_complex_tensor((batch_size, nt, nx, ny, nc), device=device) # (B, T, H, W, C)
    mask = torch.rand((batch_size, nt, 1, ny, 1), device=device) > 0.5                        # binary mask
    smaps = create_random_complex_tensor((batch_size, 1, nx, ny, nc), device=device)          # (B, 1, H, W, C)

    kspace_label = create_random_complex_tensor((batch_size, nt, nx, ny, nc), device=device)
    image_label = create_random_complex_tensor((batch_size, nt, nx, ny, 1), device=device)

    inputs = [masked_img, masked_kspace, mask, smaps]
    targets = [kspace_label, image_label]
    return inputs, targets


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = A_LIKNet(num_iter=8).to(device)
    #8次迭代

    model.eval()

    # 构造随机输入
    inputs, targets = create_inputs(device=device)

    # 推理
    with torch.no_grad():
        start = time.time()
        output_kspace, output_image = model(*inputs)
        end = time.time()

    print(f"Inference time: {end - start:.3f}s")
    print(f"Output kspace shape: {output_kspace.shape}")
    print(f"Output image shape: {output_image.shape}")