import numpy as np
import h5py
import matplotlib.pyplot as plt

from data.subsample import create_mask_for_mask_type


def compare_masks(mat_mask_path, h5_file_path, frame_index=0, acc=4, center_lines=48):
    # ========== Step 1: 读取 .mat 中的官方 mask ==========

    with h5py.File(mat_mask_path, "r") as f:
        print("Available keys in .mat file:", list(f.keys()))

        candidate_keys = ["mask04", "mask08", "mask10"]
        for key in candidate_keys:
            if key in f:
                mask_official = np.array(f[key]).astype(bool)
                break
        else:
            raise KeyError(f"No mask key found! Tried: {candidate_keys}")

    print(f"[INFO] Loaded official mask with shape: {mask_official.shape}")


    # ========== Step 2: 从 fullsample .h5 文件中读取形状 ==========
    with h5py.File(h5_file_path, "r") as f:
        kspace = f["kspace"][frame_index]  # shape: (coil, H, W)
        H, W = kspace.shape[-2], kspace.shape[-1]
        print(f"[INFO] K-space shape: ({H}, {W})")

    # ========== Step 3: 使用代码生成 mask ==========
    mask_func = create_mask_for_mask_type(
        mask_type_str="equispaced_fixed",
        center_fractions=[center_lines / W],
        accelerations=[acc],
        num_low_frequencies=[center_lines],
    )

    shape = (1, W, H)
    offset = 0
    seed = None

    mask_generated, num_low_freqs = mask_func(shape, offset, seed)  # 返回 tensor
    mask_generated = mask_generated[0][:, 0]  # shape: (W,)

    # ✅ 转为 numpy 并 cast 为 bool
    mask_generated = mask_generated.cpu().numpy().astype(bool)

    print(f"[INFO] Generated mask shape: {mask_generated.shape}")

    print("Official Mask shape:", mask_official.shape, "values:", mask_official[:,0])
    print("Generated Mask shape:", mask_generated.shape, "values:", mask_generated)


    # ========== Step 4: 比较 mask ==========
    if mask_official.shape != mask_generated.shape:
        print("[❌] Shape mismatch! Can't compare.")
        print(f"Official mask shape: {mask_official.shape}")
        print(f"Generated mask shape: {mask_generated.shape}")
        return

    diff = (mask_official != mask_generated).astype(int)
    num_diff = diff.sum()
    print(f"[RESULT] Number of different lines: {num_diff} / {H}")

    if num_diff == 0:
        print("✅ The generated mask matches the official mask.")
    else:
        print("❌ The generated mask does NOT match the official mask.")

    # ========== Step 5: 可视化 ==========
    plt.figure(figsize=(12, 3))

    plt.subplot(1, 3, 1)
    plt.plot(mask_official.astype(int), label="Official")
    plt.title("Official Mask")
    plt.xlabel("ky")

    plt.subplot(1, 3, 2)
    plt.plot(mask_generated.astype(int), label="Generated")
    plt.title("Generated Mask")
    plt.xlabel("ky")

    plt.subplot(1, 3, 3)
    plt.plot(diff, label="Difference")
    plt.title("Difference (1 = mismatch)")
    plt.xlabel("ky")

    plt.savefig("mask_compare.png", dpi=200)
    print("Saved mask comparison figure to mask_compare.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare generated and official masks")
    parser.add_argument("--mat", type=str, required=True, help="Path to .mat (v7.3 HDF5) mask file")
    parser.add_argument("--h5", type=str, required=True, help="Path to fullsample .h5 file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index")
    parser.add_argument("--acc", type=int, default=4, help="Acceleration factor")
    parser.add_argument("--center", type=int, default=48, help="Center lines count")

    args = parser.parse_args()

    compare_masks(
        mat_mask_path=args.mat,
        h5_file_path=args.h5,
        frame_index=args.frame,
        acc=args.acc,
        center_lines=args.center
    )
