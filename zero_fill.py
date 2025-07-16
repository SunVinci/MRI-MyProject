import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from data.transforms import PromptMrDataTransform
from data.subsample import create_mask_for_mask_type
from fastmri import ifft2c, rss_complex


def load_cmrxrecon_slice(file_path, frame_index):
    """
    加载 CMRxRecon 的单帧 k-space 和重建图像（从 fastMRI 格式的 h5 文件）
    """
    with h5py.File(file_path, "r") as f:
        kspace = np.array(f["kspace"])  # (num_frames, num_coils, H, W)
        target = np.array(f["reconstruction_rss"])   # (num_frames, H, W)

        kspace_frame = kspace[frame_index]  # (num_coils, H, W)

        print("kspace shape",kspace.shape)

        target_frame = target[frame_index]  # (H, W)

        attrs = {
            "max": f.attrs["max"],
            "padding_right": f.attrs["padding_right"],
            "recon_size": f.attrs["recon_size"],
        }

        return kspace_frame, target_frame, attrs, os.path.basename(file_path)


def zero_fill_reconstruct(file_path, frame_index=0, acceleration=4, center_lines=48, save_path=None):
    """
    执行单张零填充重建并可视化或保存

    Args:
        file_path: .h5 文件路径
        frame_index: 第几帧
        acceleration: 下采样倍率（如 4x）
        center_lines: 中心低频线数
        save_path: 如果设置，则保存图像到此路径
    """
    # 如果 acc==1，自动设置 center_lines 为图像宽度，避免掩码生成失败
    if acceleration == 1:
        with h5py.File(file_path, "r") as f:
            kspace_shape = f["kspace"][frame_index].shape  # (coil, H, W)
            center_lines = kspace_shape[-1]
            print(f"[INFO] acc=1 detected, setting center_lines = {center_lines} (full width)")

    # 设置 mask 函数（equispaced，固定中心）
    mask_func = create_mask_for_mask_type(
        mask_type_str="equispaced_fixed",
        center_fractions=[center_lines / 246],
        accelerations=[acceleration],
        num_low_frequencies=[center_lines],
    )

    transform = PromptMrDataTransform(mask_func=mask_func)

    # 加载数据
    kspace, target, attrs, fname = load_cmrxrecon_slice(file_path, frame_index)

    mask = np.zeros_like(kspace[0])

    # 掩膜采样并处理
    sample = transform(kspace, mask, target, attrs, fname, frame_index)

    # 零填充重建：IFFT + RSS
    image = ifft2c(sample.masked_kspace)
    recon_img = rss_complex(image).numpy()

    # 归一化显示
    recon_img = recon_img / np.max(recon_img)

    # 可视化
    plt.figure(figsize=(6, 6))
    plt.imshow(recon_img, cmap="gray")
    plt.title(f"Zero-filled Reconstruction\n{fname}, frame {frame_index}")
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CMRxRecon Zero-Filled MRI Reconstruction Demo")
    parser.add_argument("--file", type=str, required=True, help="Path to .h5 CMRxRecon file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to reconstruct")
    parser.add_argument("--acc", type=int, default=4, help="Undersampling factor")
    parser.add_argument("--center", type=int, default=48, help="Number of low-frequency center lines")
    parser.add_argument("--save", type=str, default=None, help="Save path for output image")

    args = parser.parse_args()

    zero_fill_reconstruct(
        file_path=args.file,
        frame_index=args.frame,
        acceleration=args.acc,
        center_lines=args.center,
        save_path=args.save
    )
