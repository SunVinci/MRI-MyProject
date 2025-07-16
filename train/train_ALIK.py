import os
import sys
import argparse
import logging
import torch
import gc
import psutil
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from loss import CombinedLoss

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.transforms import PromptMrDataTransform
from data.subsample import create_mask_for_mask_type
from model.ALIKNET import A_LIKNet
from pl_dataprocess.cmrxrecon_data_module import CmrxReconDataModule

def changeshape(kspace_tensor, nt, nc):
    B, nt_nc, H, W, _ = kspace_tensor.shape
    assert nt_nc == nt * nc, f"维度不一致: 期望 {nt*nc}, 实际 {nt_nc}"
    kspace_tensor = kspace_tensor.reshape(B, nt, nc, H, W, 2)
    kspace_tensor = kspace_tensor.permute(0, 1, 3, 4, 2, 5)
    return kspace_tensor

def update_mask(mask, nt):
    B, _, _, ny, _ = mask.shape
    mask = mask.repeat(1, nt, 1, 1, 1)
    return mask

def estimate_smaps(kspace: torch.Tensor, acs_size: int = 24) -> torch.Tensor:
    complex_kspace = torch.complex(kspace[..., 0], kspace[..., 1])
    B, T, H, W, C = complex_kspace.shape
    center = H // 2
    acs_start = center - acs_size // 2
    acs_end = center + acs_size // 2
    acs_kspace = complex_kspace[:, :, acs_start:acs_end, :, :]
    acs_image = torch.fft.ifft2(acs_kspace, dim=(-3, -2))
    acs_image_mean = acs_image.mean(dim=1)
    norm = torch.linalg.norm(acs_image_mean, dim=-1, keepdim=True) + 1e-8
    smaps = acs_image_mean / norm
    smaps_full = torch.zeros((B, H, W, C), dtype=smaps.dtype, device=smaps.device)
    smaps_full[:, acs_start:acs_end, :, :] = smaps
    smaps_full = smaps_full.unsqueeze(1)
    return smaps_full

def combine_coils(kspace: torch.Tensor, smaps: torch.Tensor) -> torch.Tensor:
    complex_kspace = torch.view_as_complex(kspace)
    if smaps.shape[1] == 1 and complex_kspace.shape[1] > 1:
        smaps = smaps.expand(-1, complex_kspace.shape[1], -1, -1, -1)
    coil_images = torch.fft.ifft2(complex_kspace)
    combined = (coil_images * smaps).sum(dim=-1)
    combined_realimag = torch.view_as_real(combined)
    return combined_realimag

def debug_memory(tag):
    process = psutil.Process()
    print(f"[{tag}] RAM: {process.memory_info().rss / 1024 ** 2:.2f} MB | Total Objects: {len(gc.get_objects())}")

def debug_live_tensors(label=""):
    print(f"\n🧠 [MEM DUMP] {label}")
    tensor_list = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor_list.append(obj)
        except Exception:
            continue
    print(f"🔎 Total live tensors: {len(tensor_list)}")
    from collections import Counter
    shapes = [tuple(t.shape) for t in tensor_list if hasattr(t, 'shape')]
    shape_counter = Counter(shapes)
    for shape, count in shape_counter.most_common(10):
        print(f"  🔹 Shape: {shape} x {count}")

def save_checkpoint(model, optimizer, epoch, batch_idx, out_dir="checkpoints"):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"aliknet_ep{epoch}_b{batch_idx}.pt")
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)
    print(f"📦 Checkpoint saved at {ckpt_path}")

def load_latest_checkpoint(model, optimizer, out_dir="checkpoints"):
    ckpts = sorted(Path(out_dir).glob("aliknet_ep*b*.pt"), reverse=True)
    if not ckpts:
        return 0, 0
    ckpt = torch.load(ckpts[0])
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"✅ Restored from {ckpts[0]}")
    return ckpt['epoch'], ckpt['batch_idx'] + 1


def memory_guard(model, optimizer, epoch, batch_idx, limit_gb=80):
    vm = psutil.virtual_memory()
    used_gb = (vm.total - vm.available) / 1024 ** 3  # 实际已用内存（全系统）

    if used_gb > limit_gb:
        print(f"🚨 RAM usage high: {used_gb:.2f} GB > {limit_gb} GB. Saving checkpoint and exiting...")
        save_checkpoint(model, optimizer, epoch, batch_idx)
        sys.exit(0)

def train(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    mask_func = create_mask_for_mask_type(
        args.mask_type,
        args.center_fractions,
        accelerations=args.accelerations,
        num_low_frequencies=args.num_low_frequencies,
    )

    train_transform = PromptMrDataTransform(mask_func=mask_func, use_seed=True)
    val_transform = PromptMrDataTransform(mask_func=mask_func, use_seed=True)
    test_transform = PromptMrDataTransform(mask_func=None, use_seed=True)

    data_module = CmrxReconDataModule(
        data_path=args.data_path,
        h5py_folder=args.h5py_folder,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=False,
        use_dataset_cache_file=args.use_dataset_cache_file,
    )

    data_module.prepare_data()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model = A_LIKNet(num_iter=1).to(device)
    criterion = CombinedLoss(alpha=0.8)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch, start_batch = load_latest_checkpoint(model, optimizer)
    '''
    for epoch in range(start_epoch, args.num_epochs):
        
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", unit="batch")

        for batch_idx, batch in enumerate(progress_bar):
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            nt, nc = 5, 10
            kspace = changeshape(batch.kspace, nt, nc).to(device)
            masked_kspace = changeshape(batch.masked_kspace, nt, nc).to(device)
            mask = update_mask(batch.mask, nt).to(device)
            target = batch.target.to(device)

            with torch.no_grad():
                smaps = estimate_smaps(kspace).detach()
                x_input = combine_coils(masked_kspace, smaps).detach()

            complex_kspace = torch.complex(masked_kspace[..., 0], masked_kspace[..., 1])
            output_kspace, output_image = model(x_input, complex_kspace, mask, smaps)

            if target.dim() == 5:
                target_abs = torch.abs(torch.complex(target[..., 0], target[..., 1]))
            else:
                target_abs = target

            output_abs = output_image.abs().squeeze(-1)
            loss = criterion(output_abs, target_abs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            epoch_loss += loss_value
            progress_bar.set_postfix(avg_loss=epoch_loss / (progress_bar.n + 1), current_loss=loss_value)

            output_image = output_image.detach()
            output_abs = output_abs.detach()
            del batch, kspace, masked_kspace, mask, target, smaps
            del x_input, complex_kspace, output_kspace, output_image, output_abs, target_abs, loss
            torch.cuda.empty_cache()
            gc.collect()

            if batch_idx % 10000 == 0 and batch_idx > 0:
                print(f"📦 Saving checkpoint at batch {batch_idx}...")
                save_checkpoint(model, optimizer, epoch, batch_idx)

            memory_guard(model, optimizer, epoch, batch_idx)

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}: Training loss = {avg_loss:.4f}")
        save_checkpoint(model, optimizer, epoch+1, 0)



        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                nt, nc = 5, 10
                kspace = changeshape(batch.kspace, nt, nc).to(device)
                masked_kspace = changeshape(batch.masked_kspace, nt, nc).to(device)
                mask = update_mask(batch.mask, nt).to(device)
                target = batch.target.to(device)

                smaps = estimate_smaps(kspace).detach()
                x_input = combine_coils(masked_kspace, smaps).detach()
                complex_kspace = torch.complex(masked_kspace[..., 0], masked_kspace[..., 1])
                _, output_image = model(x_input, complex_kspace, mask, smaps)

                output_abs = output_image.abs().squeeze(-1)
                if target.dim() == 5:
                    target_abs = torch.abs(torch.complex(target[..., 0], target[..., 1]))
                else:
                    target_abs = target

                val_loss = criterion(output_abs, target_abs)
                val_loss_total += val_loss.item()

                del batch, kspace, masked_kspace, mask, target, smaps
                del x_input, complex_kspace, output_image, output_abs, target_abs, val_loss
                torch.cuda.empty_cache()
                gc.collect()

        logger.info(f"Epoch {epoch + 1}: Validation loss = {val_loss_total / len(val_loader):.4f}")
    '''

    load_latest_checkpoint(model, optimizer)  # 自动恢复模型到最新

    model.eval()
    val_loss_total = 0.0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="[Validating]", unit="batch")
        for batch in progress_bar:
            nt, nc = 5, 10
            kspace = changeshape(batch.kspace, nt, nc).to(device)
            masked_kspace = changeshape(batch.masked_kspace, nt, nc).to(device)
            mask = update_mask(batch.mask, nt).to(device)
            target = batch.target.to(device)

            smaps = estimate_smaps(kspace).detach()
            x_input = combine_coils(masked_kspace, smaps).detach()
            complex_kspace = torch.complex(masked_kspace[..., 0], masked_kspace[..., 1])
            _, output_image = model(x_input, complex_kspace, mask, smaps)

            output_abs = output_image.abs().squeeze(-1)
            if target.dim() == 5:
                target_abs = torch.abs(torch.complex(target[..., 0], target[..., 1]))
            else:
                target_abs = target

            val_loss = criterion(output_abs, target_abs)
            val_loss_total += val_loss.item()

            progress_bar.set_postfix(current_loss=val_loss.item(),
                                     avg_loss=val_loss_total / (progress_bar.n + 1))

            del batch, kspace, masked_kspace, mask, target, smaps
            del x_input, complex_kspace, output_image, output_abs, target_abs, val_loss
            torch.cuda.empty_cache()
            gc.collect()

    logger.info(f"[Validation Only] Loss = {val_loss_total / len(val_loader):.4f}")

def create_arg_parser():
    parser = argparse.ArgumentParser(description="Train ALIKNET on CMRxRecon dataset")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask-type", type=str, default="equispaced_fixed")
    parser.add_argument("--accelerations", type=int, nargs="+", default=[4])
    parser.add_argument("--center-fractions", type=float, nargs="+", default=[0.08])
    parser.add_argument("--num-low-frequencies", type=int, nargs="+", default=[24])
    parser = CmrxReconDataModule.add_data_specific_args(parser)
    return parser

if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    train(args)