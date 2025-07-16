"""
metrics.py
图像重建质量评估工具集合
支持对 MRI 或医学图像的重建结果与真实图进行定量评价。
包含：MSE, NMSE, PSNR, SSIM，以及数据加载与预处理辅助函数。
"""

import numpy as np
import h5py
import math
import torch
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

###############################################
#              核心指标计算函数              #
###############################################

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """均方误差（Mean Squared Error）"""
    return np.mean((gt - pred) ** 2)

def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """归一化均方误差（Normalized MSE）"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def psnr(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """峰值信噪比（Peak Signal to Noise Ratio）"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

def ssim(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """
    结构相似性（Structural Similarity Index）
    适用于3D数据: [T, H, W]
    """
    if gt.ndim != 3 or pred.ndim != 3:
        raise ValueError("SSIM输入应为3D数组 [T, H, W]")

    maxval = maxval or gt.max()
    total = 0
    for t in range(gt.shape[0]):
        total += structural_similarity(gt[t], pred[t], data_range=maxval)
    return total / gt.shape[0]

def ssim_4d(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """
    针对4D数据 [T, Z, H, W] 计算平均 SSIM
    """
    if gt.ndim != 4 or pred.ndim != 4:
        raise ValueError("SSIM_4D 输入应为4D数组 [T, Z, H, W]")

    maxval = maxval or gt.max()
    total = 0
    for t in range(gt.shape[0]):
        total += ssim(gt[t], pred[t], maxval=maxval)
    return total / gt.shape[0]

def cal_metric(gt, pred):
    """
    计算一组完整的评价指标（NMSE, PSNR, SSIM）
    输入：gt, pred 均为 np.ndarray，取模后传入
    """
    return nmse(gt, pred), psnr(gt, pred), ssim_4d(gt, pred)


###############################################
#                .mat 数据读取                #
###############################################

def loadmat(filename):
    """加载 Matlab v7.3 .mat 文件（支持嵌套结构）"""
    with h5py.File(filename, 'r') as f:
        return _loadmat_group(f)

def _loadmat_group(group):
    data = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            data[k] = v[()]
        elif isinstance(v, h5py.Group):
            data[k] = _loadmat_group(v)
    return data

def load_kdata(filename):
    """
    从.mat文件读取复数k空间数据，返回 [T, Z, C, H, W] 结构
    """
    data = loadmat(filename)
    key = list(data.keys())[0]  # 默认取第一个字段
    kdata = data[key]['real'] + 1j * data[key]['imag']
    return kdata


def extract_number(filename):
    """从文件名中提取数字字符串（排序或匹配用）"""
    return ''.join(filter(str.isdigit, filename))


###############################################
#              数据增强与裁剪辅助             #
###############################################

def matlab_round(n):
    return int(n + 0.5) if n > 0 else int(n - 0.5)

def _crop(a, crop_shape):
    """中心裁剪至指定形状"""
    indices = [
        (math.floor(dim / 2) + math.ceil(-crop_dim / 2),
         math.floor(dim / 2) + math.ceil(crop_dim / 2))
        for dim, crop_dim in zip(a.shape, crop_shape)
    ]
    return a[indices[0][0]:indices[0][1],
             indices[1][0]:indices[1][1],
             indices[2][0]:indices[2][1],
             indices[3][0]:indices[3][1]]

def crop_submission(a, ismap=False):
    """
    为比赛格式或统一大小进行裁剪，支持3/2缩放比例
    a: ndarray, 形状 [X, Y, Z, T]
    ismap: bool, 是否为概率图
    """
    sx, sy, sz, st = a.shape
    if sz >= 3:
        a = a[:, :, matlab_round(sz / 2) - 2:matlab_round(sz / 2)]

    if ismap:
        return _crop(a, (matlab_round(sx / 3), matlab_round(sy / 2), 2, st))
    else:
        return _crop(a[..., 0:3], (matlab_round(sx / 3), matlab_round(sy / 2), 2, 3))


###############################################
#            简单图像增强（旋转翻转）          #
###############################################

def rotate(image, mode):
    """
    数据增强：对图像进行不同方式的旋转或翻转（适用于 numpy 和 torch）
    """
    if mode == 0:
        return image
    elif mode == 1:
        return torch.flip(image, [2])
    elif mode == 2:
        return np.rot90(image)
    elif mode == 3:
        return np.flipud(np.rot90(image))
    elif mode == 4:
        return torch.rot90(image, k=2, dims=[2, 3])
    elif mode == 5:
        return torch.flip(torch.rot90(image, k=2, dims=[2, 3]), [2])
    elif mode == 6:
        return np.rot90(image, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(image, k=3))
    else:
        raise ValueError('Invalid augmentation mode')

def rotate_re(image, mode):
    """反向恢复旋转或翻转操作"""
    if mode == 0:
        return image
    elif mode == 1:
        return torch.flip(image, [2])
    elif mode == 2:
        return torch.rot90(image, k=-1)
    elif mode == 3:
        return np.rot90(np.flipud(image), k=-1)
    elif mode == 4:
        return torch.rot90(image, k=-2, dims=[2, 3])
    elif mode == 5:
        return torch.rot90(torch.flip(image, [2]), k=-2, dims=[2, 3])
    elif mode == 6:
        return np.rot90(image, k=-3)
    elif mode == 7:
        return np.rot90(np.flipud(image), k=-3)
    else:
        raise ValueError('Invalid reverse mode')


###############################################
#              模型参数统计工具              #
###############################################

def count_parameters(model):
    """统计模型的总参数量"""
    return sum(p.numel() for p in model.parameters()) if model else 0

def count_trainable_parameters(model):
    """统计模型中可训练的参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) if model else 0

def count_untrainable_parameters(model):
    """统计模型中冻结的参数数量"""
    return sum(p.numel() for p in model.parameters() if not p.requires_grad) if model else 0
