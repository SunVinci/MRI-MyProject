import os
import sys

# 获取 train_ALIK.py 所在目录
current_dir = os.path.dirname(__file__)

# 获取项目根目录（train 上一层）
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# 添加项目根目录到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import logging
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pathlib import Path

from fastmri.data import SliceDataset
from data.transforms import PromptMrDataTransform
from data.subsample import create_mask_for_mask_type
from model.ALIKNET import A_LIKNet

def train(args):
    # ---------------------------
    # Logging & Device Setup
    # ---------------------------
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ---------------------------
    # Create Mask Function
    # ---------------------------
    mask_func = create_mask_for_mask_type(
        args.mask_type,
        center_fractions=args.center_fractions,
        accelerations=args.accelerations,
        num_low_frequencies=args.num_low_frequencies
    )

    # ---------------------------
    # Transforms
    # ---------------------------
    train_transform = PromptMrDataTransform(mask_func=mask_func, use_seed=True)
    val_transform = PromptMrDataTransform(mask_func=None, use_seed=True)

    # ---------------------------
    # Datasets
    # ---------------------------
    train_data = SliceDataset(
        root=args.data_path / "train",
        transform=train_transform,
        challenge="multicoil"
    )
    val_data = SliceDataset(
        root=args.data_path / "val",
        transform=val_transform,
        challenge="multicoil"
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2)

    logger.info(f"Loaded {len(train_loader)} training batches and {len(val_loader)} validation batches.")

    # ---------------------------
    # Model Placeholder
    # ---------------------------

    model = A_LIKNet(num_iter=8).to(device)

    # ---------------------------
    # Loss & Optimizer
    # ---------------------------
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_loader:
            # -----------------------------
            # 获取数据并放入设备
            # -----------------------------
            masked_kspace = batch.masked_kspace.to(device)  # [B, T, H, W, 2]
            mask = batch.mask.to(device)  # [B, T, H, W, 1]
            smaps = batch.smaps.to(device)  # [B, T, H, W, 2]（你需要确保 dataloader 输出它）
            target = batch.target.to(device)  # [B, T, H, W] or [B, T, H, W, 2]

            # -----------------------------
            # 构建输入：zero-filled 图像
            # -----------------------------
            # 将 masked_kspace 从 real+imag 转为复数
            complex_kspace = torch.complex(masked_kspace[..., 0], masked_kspace[..., 1])  # [B, T, H, W]
            zero_filled = torch.fft.ifft2(complex_kspace)  # [B, T, H, W]
            zero_filled_realimag = torch.view_as_real(zero_filled)  # [B, T, H, W, 2]

            # -----------------------------
            # 模型推理
            # -----------------------------
            output_kspace, output_image = model(zero_filled_realimag, complex_kspace, mask,
                                                smaps)  # image是 [B, T, H, W, 2]

            # -----------------------------
            # 损失函数计算
            # -----------------------------
            output_abs = torch.abs(torch.complex(output_image[..., 0], output_image[..., 1]))  # [B, T, H, W]
            if target.dim() == 5:  # target 可能是复数
                target_abs = torch.abs(torch.complex(target[..., 0], target[..., 1]))
            else:
                target_abs = target

            loss = criterion(output_abs, target_abs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(f"Epoch {epoch}: Training loss = {loss.item():.4f}")
        # TODO: Validation and save

def create_arg_parser():
    parser = argparse.ArgumentParser(description="Train ALIKNET on CMRxRecon dataset")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to dataset root")
    parser.add_argument("--mask-type", type=str, default="equispaced_fixed")
    parser.add_argument("--center-fractions", type=float, nargs="+", default=[0.08])   #保留中心部分低频比例
    parser.add_argument("--num-low-frequencies", type=int, default=24)     #明确指定中心部分保留列数
    parser.add_argument("--accelerations", type=int, nargs="+", default=[4])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser

if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    train(args)