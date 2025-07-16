import torch
import torch.nn as nn
import torch.nn.functional as F

class LNetXYTBatch(nn.Module):
    def __init__(self, patch_size=(32, 32), max_batch_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.max_batch_size = max_batch_size
        self.min_sigma = 1e-6

        self.thres_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def extract_patches(self, x):
        # x: [B, T, H, W] complex
        B, T, H, W = x.shape
        Px, Py = self.patch_size

        pad_h = (Px - H % Px) % Px
        pad_w = (Py - W % Py) % Py

        x_padded = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
        H_pad, W_pad = x_padded.shape[-2:]

        self.patch_info = {
            'H': H, 'W': W,
            'H_pad': H_pad, 'W_pad': W_pad,
            'pad_h': pad_h, 'pad_w': pad_w,
            'Px': Px, 'Py': Py,
        }

        x_reshaped = x_padded.reshape(B * T, 1, H_pad, W_pad)
        patches = x_reshaped.unfold(2, Px, Px).unfold(3, Py, Py)
        patches = patches.contiguous().view(B, T, -1, Px, Py)
        return patches

    def recover_patches(self, x):
        # x: [B, T, N, Px, Py] complex
        B, T, N, Px, Py = x.shape
        info = self.patch_info
        num_patches_x = info['H_pad'] // Px
        num_patches_y = info['W_pad'] // Py
        assert N == num_patches_x * num_patches_y

        x = x.view(B, T, num_patches_x, num_patches_y, Px, Py)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, T, num_patches_x * Px, num_patches_y * Py)
        return x[:, :, :info['H'], :info['W']]

    def low_rank(self, L_complex):
        nb, nt, n = L_complex.shape
        q_batches = []

        for i in range(0, nb, self.max_batch_size):
            L_batch = L_complex[i:i + self.max_batch_size]
            L_mag = torch.abs(L_batch)

            U, S, Vh = torch.linalg.svd(L_mag, full_matrices=False)
            s0 = S[:, 0:1]
            thres_factor = self.thres_mlp(s0)
            thres = thres_factor * s0

            S = torch.clamp(S, min=self.min_sigma)
            S_thresh = F.relu(S - thres) + thres * torch.sigmoid(S - thres)

            US = U * S_thresh.unsqueeze(2)
            L_mag_recon = torch.bmm(US, Vh)

            phase = torch.angle(L_batch)
            L_recon = L_mag_recon * torch.exp(1j * phase)

            q_batches.append(L_recon)

            del L_batch, L_mag, U, S, Vh, US, L_mag_recon, phase, L_recon
            torch.cuda.empty_cache()

        return torch.cat(q_batches, dim=0)

    def forward(self, x):
        """
        x: Tensor of shape (B, T, H, W, 2) or (B, T, H, W) complex
        """
        is_realimag = False
        if x.ndim == 5:
            x = torch.complex(x[..., 0], x[..., 1])
            is_realimag = True

        B, T, H, W = x.shape
        patches = self.extract_patches(x)  # (B, T, N, Px, Py)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # (B, N, T, Px, Py)
        B, N, T, Px, Py = patches.shape
        patches = patches.view(B * N, T, Px * Py)

        patches_recon = self.low_rank(patches)
        patches_recon = patches_recon.view(B, N, T, Px, Py).permute(0, 2, 1, 3, 4)  # (B, T, N, Px, Py)

        x_recon = self.recover_patches(patches_recon)

        if is_realimag:
            return torch.stack([x_recon.real, x_recon.imag], dim=-1)
        return x_recon
