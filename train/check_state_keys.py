import torch
import os
import sys

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.ALIKNET import A_LIKNet

def compare_state_dict_keys(checkpoint_path):
    # 加载 checkpoint（只加载参数字典，不载入模型）
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ckpt_keys = set(checkpoint['model_state_dict'].keys())

    # 实例化当前模型，获取当前模型参数名
    model = A_LIKNet(num_iter=1)  # 按需改模型构造参数
    model_keys = set(model.state_dict().keys())

    # 差集分析
    only_in_ckpt = ckpt_keys - model_keys
    only_in_model = model_keys - ckpt_keys
    common_keys = ckpt_keys & model_keys

    print(f"参数数目: checkpoint={len(ckpt_keys)}, model={len(model_keys)}")
    print(f"只在 checkpoint 中的参数 ({len(only_in_ckpt)}) ：")
    for k in sorted(only_in_ckpt):
        print(f"  {k}")
    print(f"\n只在当前模型中的参数 ({len(only_in_model)}) ：")
    for k in sorted(only_in_model):
        print(f"  {k}")
    print(f"\n共有的参数 ({len(common_keys)}) ：示例（最多10个）")
    for k in list(sorted(common_keys))[:10]:
        print(f"  {k}")

if __name__ == "__main__":
    checkpoint_path = "checkpoints/aliknet_ep0_b46.pt"  # 改成你自己的路径
    compare_state_dict_keys(checkpoint_path)
