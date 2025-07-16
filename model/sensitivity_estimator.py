
import torch
from torch import nn
from typing import List, Optional
import fastmri

# 导入以下模块：
# - SensitivityModel
# - PromptMRBlock
# - NormPromptUnet

class SensitivityEstimator(nn.Module):
    def __init__(
        self,
        num_adj_slices: int = 5,
        in_chans: int = 2,
        out_chans: int = 2,
        n_feat0: int = 24,
        feature_dim: List[int] = [36, 48, 60],
        prompt_dim: List[int] = [12, 24, 36],
        len_prompt: List[int] = [5, 5, 5],
        prompt_size: List[int] = [64, 32, 16],
        n_enc_cab: List[int] = [2, 3, 3],
        n_dec_cab: List[int] = [2, 2, 3],
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        mask_center: bool = True,
        low_mem: bool = False,
    ):
        super().__init__()
        self.model = SensitivityModel(
            in_chans=in_chans,
            out_chans=out_chans,
            num_adj_slices=num_adj_slices,
            n_feat0=n_feat0,
            feature_dim=feature_dim,
            prompt_dim=prompt_dim,
            len_prompt=len_prompt,
            prompt_size=prompt_size,
            n_enc_cab=n_enc_cab,
            n_dec_cab=n_dec_cab,
            n_skip_cab=n_skip_cab,
            n_bottleneck_cab=n_bottleneck_cab,
            no_use_ca=no_use_ca,
            mask_center=mask_center,
            low_mem=low_mem,
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        return self.model(masked_kspace, mask, num_low_frequencies)


class PromptMR(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        num_adj_slices: int = 5,
        n_feat0: int = 48,
        feature_dim: List[int] = [72, 96, 120],
        prompt_dim: List[int] = [24, 48, 72],
        len_prompt: List[int] = [5, 5, 5],
        prompt_size: List[int] = [64, 32, 16],
        n_enc_cab: List[int] = [2, 3, 3],
        n_dec_cab: List[int] = [2, 2, 3],
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        sensitivity_estimator: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        assert num_adj_slices % 2 == 1, "num_adj_slices must be odd"
        self.num_adj_slices = num_adj_slices
        self.center_slice = num_adj_slices // 2
        self.use_checkpoint = use_checkpoint

        if sensitivity_estimator is None:
            raise ValueError("Must provide a SensitivityEstimator instance.")
        self.sensitivity_estimator = sensitivity_estimator

        self.cascades = nn.ModuleList([
            PromptMRBlock(
                NormPromptUnet(
                    in_chans=2 * num_adj_slices,
                    out_chans=2 * num_adj_slices,
                    n_feat0=n_feat0,
                    feature_dim=feature_dim,
                    prompt_dim=prompt_dim,
                    len_prompt=len_prompt,
                    prompt_size=prompt_size,
                    n_enc_cab=n_enc_cab,
                    n_dec_cab=n_dec_cab,
                    n_skip_cab=n_skip_cab,
                    n_bottleneck_cab=n_bottleneck_cab,
                    no_use_ca=no_use_ca,
                ),
                num_adj_slices=num_adj_slices,
            )
            for _ in range(num_cascades)
        ])

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:

        if self.use_checkpoint and self.training:
            sens_maps = torch.utils.checkpoint.checkpoint(
                self.sensitivity_estimator, masked_kspace, mask, num_low_frequencies, use_reentrant=False)
        else:
            sens_maps = self.sensitivity_estimator(
                masked_kspace, mask, num_low_frequencies)

        kspace_pred = masked_kspace.clone()
        for cascade in self.cascades:
            if self.use_checkpoint and self.training:
                kspace_pred = torch.utils.checkpoint.checkpoint(
                    cascade, kspace_pred, masked_kspace, mask, sens_maps, use_reentrant=False)
            else:
                kspace_pred = cascade(
                    kspace_pred, masked_kspace, mask, sens_maps)

        kspace_pred = torch.chunk(kspace_pred, self.num_adj_slices, dim=1)[
            self.center_slice]

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
