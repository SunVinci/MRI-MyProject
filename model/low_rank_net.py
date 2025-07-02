import torch
import torch.nn as nn
import torch.nn.functional as F

class LNetXYTBatch(nn.Module):
    def __init__(self, num_patches):
        super().__init__()
        self.num_patches = num_patches
        self.thres_coef = nn.Parameter(torch.full((num_patches,), -2.0, dtype=torch.float32))
        self.min_sigma = 1e-6

    def low_rank(self, L):

        nb, nt, n = L.shape
        U, S, Vh = torch.linalg.svd(L, full_matrices=False)  # Vh: (nb, nt, n)


        if self.thres_coef.size(0) != nb:
            if self.thres_coef.size(0) > nb:
                thres_coef = self.thres_coef[:nb]
            else:
                new_coef = torch.full((nb - self.thres_coef.size(0),), -2.0,
                                      dtype=self.thres_coef.dtype, device=self.thres_coef.device)
                thres_coef = torch.cat([self.thres_coef, new_coef])
                self.thres_coef = nn.Parameter(thres_coef)
        else:
            thres_coef = self.thres_coef

        thres = torch.sigmoid(thres_coef) * S[:, 0]  # (nb,)
        thres = thres.unsqueeze(1)  # (nb, 1)


        S = torch.clamp(S, min=self.min_sigma)
        S_thresh = F.relu(S - thres) + thres * torch.sigmoid(S - thres)  # (nb, nt)

        US = U * S_thresh.unsqueeze(2)  # (nb, nt, nt)
        #print(f"US shape: {US.shape}")  # 应输出 (24, 25, 25)

        #print(f"Vh shape: {Vh.shape}")  # 应输出 (24, 25, 1024)

        L_recon = torch.bmm(US, Vh)  # (nb, nt, n)
        #print(f"L_recon shape: {L_recon.shape}")  # 应输出 (24, 25, 1024)
        return L_recon

    def forward(self, x):
        # x: (nb, nt, nx, ny) or (nb, nt, nx, ny, 1)
        squeeze_last = False
        if x.ndim == 5:
            x = x.squeeze(-1)
            squeeze_last = True

        nb, nt, nx, ny = x.shape
        L_flat = x.view(nb, nt, nx * ny)  # (nb, nt, nx*ny)
        L_recon = self.low_rank(L_flat)  # (nb, nt, nx*ny)
        L_out = L_recon.view(nb, nt, nx, ny)  # (nb, nt, nx, ny)

        if squeeze_last:
            L_out = L_out.unsqueeze(-1)  # (nb, nt, nx, ny, 1)

        return L_out