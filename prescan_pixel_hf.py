"""prescan_pixel_hf.py — 【自包含裁剪版】仅保留 radial_band_masks_2d。

原文件位于 wan_bench/prescan_pixel_hf.py(146 行, 像素级高频差距诊断脚本), 其模块级
`from wanbench.generation import ...` 会拖入 wan_mhfno/wan_common 依赖链; 本仓库的
noise_struct.py 只用到其中的 radial_band_masks_2d(纯 torch.fft, 无外部依赖), 故逐字
保留该函数, 其余部分与 λ 扫描实验无关。函数体与原文件逐字一致。
"""
import torch


def radial_band_masks_2d(H, W, K=8, device=None):
    """2D rFFT 径向频率 → K 个硬掩码 [K, H, W//2+1]"""
    fh = torch.fft.fftfreq(H, device=device).abs()
    fw = torch.fft.rfftfreq(W, device=device).abs()
    gh, gw = torch.meshgrid(fh, fw, indexing="ij")
    radius = torch.sqrt(gh ** 2 + gw ** 2)
    edges = torch.linspace(0, radius.max().item(), K + 1, device=device)
    return torch.stack([((radius >= edges[k]) & (radius < edges[k + 1])).float()
                        for k in range(K)])
