"""ll_blend.py — 剧本混合(2026-10-02): 把新视频的初始噪声朝"目标剧本 seed"连续插值。

数学: ε' = (ε_b + λ(ε_d − ε_b)) / √((1−λ)² + λ²)   ← 分母保持 std=1(白性)
  band='field': 整场混合(全部系数)
  band='ll'   : 只混合 LL 带(帧 1..T-1, 逐帧 wavedec, LL 混合+带内重归一, 细节带=base)
帧 0 恒不触(I2V 中该槽被条件图占据)。
"""
import math

import numpy as np
import torch


def lp_mask(H, W, frac):
    """rFFT 低通 mask: 径向频率升序累计**实自由度**至 frac·H·W(自共轭列计 1, 其余计 2)。

    白噪声下正交系数均方相等 ⇒ frac 为精确的期望能量分数与实 dof 分数(引理 1 子空间维度);
    frac=0.25 ⇒ 390 实 dof/帧 = Haar LL1(15×26)同维度, 两模式可比(预注册 P10)。
    注意: 非自共轭复系数携带 2 个实自由度, 按"存储系数个数"选会使能量分数≈2×frac。
    """
    if not 0 < frac <= 1:
        raise ValueError(f"frac 需 ∈(0,1], got {frac}")
    ky = np.fft.fftfreq(H)
    kx = np.fft.rfftfreq(W)
    selfconj = np.zeros((H, kx.size), bool)
    selfconj[:, 0] = True
    if W % 2 == 0:
        selfconj[:, -1] = True          # kx = Nyquist 列, 实系数
    dof = np.where(selfconj, 1, 2).ravel()
    r = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).ravel()
    target = int(round(frac * H * W))
    m = np.zeros(r.size, bool)
    acc = 0
    for idx in np.argsort(r, kind="stable"):
        if acc >= target:
            break
        m[idx] = True
        acc += int(dof[idx])
    if acc < target:
        raise ValueError(f"dof 预算 {target} 超出可实现上限 {acc}")
    return m.reshape(H, kx.size), acc


def blend_eps(eps_base, eps_donor, lam, band="field", level=1, wave="haar", frac=0.25):
    """返回混合后的噪声场(torch, 同形)。
    λ=0 ⇒ base 逐位; field λ=1 ⇒ donor 逐位;
    band 模式 λ=1 ⇒ 带替换(补带保持 base)。"""
    assert 0.0 <= lam <= 1.0
    if lam == 0.0:
        return eps_base.clone()
    if lam == 1.0 and band == "field":
        return eps_donor.clone()          # 浮点上 b+(d−b)≠d 逐位, 特判保证端点精确
    b = eps_base.clone().float()
    d = eps_donor.clone().float()
    _, T, H, W = b.shape
    norm = math.sqrt((1 - lam) ** 2 + lam ** 2)
    if band == "field":
        out = (b + lam * (d - b)) / norm
    elif band == "lp":
        m, _ = lp_mask(H, W, frac)
        fb = np.fft.rfft2(b.numpy(), axes=(-2, -1), norm="ortho")
        fd = np.fft.rfft2(d.numpy(), axes=(-2, -1), norm="ortho")
        fb[..., m] = (fb[..., m] + lam * (fd[..., m] - fb[..., m])) / norm
        rec = np.fft.irfft2(fb, s=(H, W), axes=(-2, -1), norm="ortho")
        out = torch.from_numpy(rec.astype(np.float32))
    elif band == "hh":
        import pywt
        out = b.clone()
        e = b.numpy(); dn = d.numpy()
        C = b.shape[0]
        for c in range(C):
            for t in range(1, T):
                cb = pywt.wavedec2(e[c, t], wave, mode="periodization", level=level)
                cd = pywt.wavedec2(dn[c, t], wave, mode="periodization", level=level)
                for k in range(1, len(cb)):              # 只混细节带, LL=base(刹车臂)
                    cb[k] = tuple((a + lam * (da - a)) / norm for a, da in zip(cb[k], cd[k]))
                rec = pywt.waverec2(cb, wave, mode="periodization")
                out[c, t] = torch.from_numpy(rec[:H, :W].astype(np.float32))
    elif band == "ll":
        import pywt
        out = b.clone()
        e = b.numpy(); dn = d.numpy()
        C = b.shape[0]
        for c in range(C):
            for t in range(1, T):
                cb = pywt.wavedec2(e[c, t], wave, mode="periodization", level=level)
                cd = pywt.wavedec2(dn[c, t], wave, mode="periodization", level=level)
                cb[0][:] = (cb[0] + lam * (cd[0] - cb[0])) / norm
                rec = pywt.waverec2(cb, wave, mode="periodization")
                out[c, t] = torch.from_numpy(rec[:H, :W].astype(np.float32))
    else:
        raise KeyError(band)
    out[:, 0] = eps_base[:, 0].float()          # 帧 0 恒不触
    return out.to(eps_base.dtype)
