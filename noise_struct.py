"""noise_struct.py — 噪声构造器库 (设计文档 §5.2, PLAN_noise_seed_v1 T1.4)

三维度算子 (作用于 [C,T,H,W] 噪声张量):
  D1 bandshare(eps_a, eps_fresh, k)      空间径向带 0~k 取 eps_a, 其余取 fresh
  D2 reuse_shuffle(eps_a, S, seed)       帧袋复用 + 局部 shuffle + 边界锚定
  D3 advect(eps, flow, seed)             整值散射平流 (无插值, 保白性)
  blend_gamma(s, z, gamma)               结构:新鲜 混合, 重归一化保单位方差

白性工具: morans_i / ks_white (GwtF 教训: 该白的地方必须白)。
全部纯函数 CPU 可测; 测试 tests/test_noise_struct.py。
"""
import torch
import torch.nn.functional as F

from prescan_pixel_hf import radial_band_masks_2d  # noqa: E402  (8 径向带, 母文档口径)


# ==========================================
# 白性检验
# ==========================================

def morans_i(field):
    """[C,T,H,W] → 空间自相关 (Moran's I)。iid ≈ 0; 平滑场 → →1。

    对每 (c,t) 帧计算 4 邻接 Moran's I 后取均值:
    I = (n/S0)·Σ_ij w_ij z_i z_j / Σ_i z_i²,  z = x − mean(x)
    """
    x = field.detach().float()
    C, T, H, W = x.shape
    z = x - x.mean(dim=(-2, -1), keepdim=True)
    num = torch.zeros(())
    den = torch.zeros(())
    for c in range(C):
        for t in range(T):
            zt = z[c, t]
            prod = (zt[:-1, :] * zt[1:, :]).sum() + (zt[:, :-1] * zt[:, 1:]).sum()
            num = num + 2.0 * prod                      # 每条邻接边计两次 (对称)
            den = den + (zt * zt).sum()
    n = H * W
    s0 = 2.0 * (n - H) + 2.0 * (n - W)                  # 邻接边权重和 (对称计数)
    return float(n / s0 * num / (den + 1e-12)) if den > 0 else 0.0


def ks_white(field, n_sample=4000, alpha=0.01):
    """K-S 检验边缘分布 ~ N(0,1)。真白噪声不应被拒绝 (alpha=0.01)。
    下采样用独立确定性 generator (不依赖全局 RNG 状态 → 测试可复现)。"""
    from scipy import stats
    v = field.detach().float().flatten()
    if v.numel() > n_sample:
        g = torch.Generator().manual_seed(20260905)
        idx = torch.randint(0, v.numel(), (n_sample,), generator=g)
        v = v[idx]
    stat, p = stats.kstest(v.numpy(), "norm")
    return p > alpha


# ==========================================
# D1 空间径向带共享
# ==========================================

def bandshare(eps_a, eps_fresh, k, K=8):
    """逐帧 2D rFFT: 空间径向带 0~k 系数取 eps_a, 带 k+1~K-1 取 eps_fresh。

    逐频系数整体取自单一源 (无插值/无缩放) → 白性与方差按构造保持。
    k=K-1 时逐位等于 eps_a (全保留恒等旁路, 不做重归一化)。
    """
    assert eps_a.shape == eps_fresh.shape
    if k >= K - 1:
        return eps_a.clone()
    C, T, H, W = eps_a.shape
    keep = radial_band_masks_2d(H, W, K=K)              # [K, H, W//2+1]
    mask_keep = torch.stack([keep[b] for b in range(k + 1)]).sum(dim=0).bool()
    Fa = torch.fft.rfftn(eps_a.float(), dim=(-2, -1))
    Ff = torch.fft.rfftn(eps_fresh.float(), dim=(-2, -1))
    F = torch.where(mask_keep, Fa, Ff)
    out = torch.fft.irfftn(F, s=(H, W), dim=(-2, -1))
    # irfftn 数值误差累积可致方差轻微偏离; 按帧重归一化到单位方差
    var = out.var(dim=(-2, -1), keepdim=True)
    out = out / (var + 1e-8).sqrt()
    return out.to(eps_a.dtype)


# ==========================================
# D2 时间帧袋复用 + 局部 shuffle + 边界锚定
# ==========================================

def reuse_shuffle(eps_a, S, seed, shuffle_prob=1.0):
    """[C,T,H,W] → 同形帧袋重排 (复用, 不创造新帧值)。

    - 输出前 S 帧 = eps_a 末 S 帧 (原序, 边界锚定);
    - 其余帧: 剩余帧袋按 S 分块循环取用, 每块内部以 shuffle_prob 概率洗牌;
    - shuffle_prob=0 → 纯循环复用 (FreeNoise 常数极端)。
    """
    C, T, H, W = eps_a.shape
    g = torch.Generator().manual_seed(seed)
    idx_anchor = list(range(T - S, T))                  # 末 S 帧原序
    rest = list(range(T - S))                           # 剩余帧袋
    blocks = [rest[i:i + S] for i in range(0, len(rest), S)]
    out_idx = list(idx_anchor)
    for blk in blocks:
        blk = list(blk)
        if torch.rand(1, generator=g).item() < shuffle_prob and len(blk) > 1:
            perm = torch.randperm(len(blk), generator=g).tolist()
            blk = [blk[j] for j in perm]
        out_idx.extend(blk)
    out_idx = out_idx[:T]
    return eps_a[:, torch.tensor(out_idx, dtype=torch.long)]


# ==========================================
# D3 整值散射平流 (GwtF Algorithm 1 简化版)
# ==========================================

def advect(eps, flow, seed):
    """eps [C,T,H,W]; flow [T-1,H,W,2] 整值 (dy,dx) 网格位移 → 同形输出。

    语义 (GwtF 原则的整值散射版; 起点=eps[:,0], 逐帧向前输运):
      - frame t+1 的目标格 (y',x') 收到所有 (y+dy,x+dx)=(y',x') 的源格值,
        多源之和除 √计数 重归一化 (保单位方差); 空目标格填新鲜 N(0,1);
      - 空间边界: 环绕 (噪声场平移不变, 环绕统计等价于出界+补新);
      - 无插值 → 每帧边缘分布仍白; 零流 → 帧恒定 (= D2 常数极端)。
    """
    C, T, H, W = eps.shape
    g = torch.Generator().manual_seed(seed)
    out = torch.empty_like(eps)
    out[:, 0] = eps[:, 0]
    cur = eps[:, 0]
    for t in range(T - 1):
        fy = flow[t, :, :, 0].reshape(-1)
        fx = flow[t, :, :, 1].reshape(-1)
        yy = (torch.arange(H).view(-1, 1).expand(H, W).reshape(-1) + fy) % H
        xx = (torch.arange(W).view(1, -1).expand(H, W).reshape(-1) + fx) % W
        tgt = yy * W + xx
        flat = cur.reshape(C, -1)
        acc = torch.zeros(C, H * W, dtype=eps.dtype)
        cnt = torch.zeros(H * W)
        acc.index_add_(1, tgt, flat)
        cnt.index_add_(0, tgt, torch.ones_like(tgt, dtype=cnt.dtype))
        fresh = torch.randn(C, H * W, generator=g)
        mask = cnt > 0
        acc_n = acc / cnt.clamp(min=1).unsqueeze(0).sqrt()
        cur = torch.where(mask.unsqueeze(0), acc_n, fresh).reshape(C, H, W)
        out[:, t + 1] = cur
    return out


# ==========================================
# γ 混合
# ==========================================

def blend_gamma(s, z, gamma):
    """(1-γ)·s + γ·z, 再除 √((1-γ)²+γ²) 保单位方差 (GwtF 退化式)"""
    out = (1.0 - gamma) * s.float() + gamma * z.float()
    return (out / ((1.0 - gamma) ** 2 + gamma ** 2) ** 0.5).to(s.dtype)


# ==========================================
# W 合成井注入 (PLAN_well_inject_v2 M0a, 文献精读 2026-09-10)
#   构造器 A well_inject   能量塑形: out = (eps + E·low_{≤k}(eps)) / 逐帧方差
#   构造器 B patch_repeat  时序相关注入 (FreeTraj Eq.8 频带版, 第四类构造):
#                          边缘白性精确保持, 框内跨帧相关=1
#   控制臂  diffuse_boost  能量匹配弥散 (H2: 定位 vs 能量归因)
# ==========================================

def low_pass_2d(eps, k, K=8):
    """逐帧 2D rFFT 径向带 0~k 低通分量 (bandshare 同款掩码, 不重归一化)。"""
    _, _, H, W = eps.shape
    keep = radial_band_masks_2d(H, W, K=K)
    mask = torch.stack([keep[b] for b in range(k + 1)]).sum(dim=0).bool()
    F = torch.fft.rfftn(eps.float(), dim=(-2, -1))
    return torch.fft.irfftn(F * mask, s=(H, W), dim=(-2, -1))


def well_envelope(H, W, wells):
    """井列表 [(y0,x0,r,amp),...] → 空间包络 [H,W]。逐井 exp(-d²/2r²)·amp 求和。
    时间恒定 (14.7: 可读性需要时序相干 — 位置信号不随帧抖动)。"""
    yy = torch.arange(H).view(-1, 1).float()
    xx = torch.arange(W).view(1, -1).float()
    E = torch.zeros(H, W)
    for (y0, x0, r, amp) in wells:
        E = E + amp * torch.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2.0 * r * r))
    return E


def well_inject(eps, wells, k=1, K=8, temporal_mask=None):
    """构造器 A · 能量塑形: out = (eps + E·low_{≤k}(eps)) / 逐帧方差。

    线性作用于高斯 → 联合高斯性保持 (区别于值替换类); E 时间恒定, low 逐帧独立。
    逐帧方差重归一化 = 能量再平衡 (ϕ-Noise 教训: 剂量×再平衡决定分布内程度)。
    amp 全零 → 逐位恒等 (clone 旁路, 测试锚点)。
    temporal_mask: [T] 0/1 张量, 逐帧门控包络 (wellSparse: 间歇等剂量设计,
    破坏时间相干而保持时间平均能量 — 与全帧 well 的载体判别对照)。
    """
    if all(w[3] == 0 for w in wells):
        return eps.clone()
    _, _, H, W = eps.shape
    E = well_envelope(H, W, wells)
    low = low_pass_2d(eps, k, K)
    mod = E[None, None]
    if temporal_mask is not None:
        mod = mod * temporal_mask.float()[None, :, None, None]   # [1,T,1,1]
    out = eps.float() + mod * low
    var = out.var(dim=(-2, -1), keepdim=True)
    scale = (var + 1e-8).sqrt()
    if temporal_mask is not None:
        # 只重归一激活帧: 非激活帧保持逐位基线 (sparse 的零修改语义)
        scale = 1.0 + temporal_mask.float()[None, :, None, None] * (scale - 1.0)
    out = out / scale
    return out.to(eps.dtype)


def diffuse_boost(eps, wells, k=1, K=8):
    """能量匹配弥散控制臂 (H2): E 取逐 (c,t) 常数 c_f, 使 ‖修正‖ 与 well_inject
    同实现逐帧相等 → 若弥散臂也产生方向性位移, 位移来自低频能量而非空间定位。"""
    _, _, H, W = eps.shape
    E = well_envelope(H, W, wells)
    low = low_pass_2d(eps, k, K)
    mod_well = E[None, None] * low
    n_well = mod_well.flatten(2).norm(dim=2)            # [C,T]
    n_low = low.flatten(2).norm(dim=2)
    c = n_well / (n_low + 1e-12)
    out = eps.float() + c[:, :, None, None] * low
    var = out.var(dim=(-2, -1), keepdim=True)
    out = out / (var + 1e-8).sqrt()
    return out.to(eps.dtype)


def patch_repeat(eps, boxes, k=1, K=8, seed=0):
    """构造器 B · 时序相关注入 (FreeTraj Eq.8 频带版)。

    boxes=[(y0,x0,h,w),...]: 逐框采样独立白噪声块 [C,1,h,w], 跨帧写入框内
    (框内时序相关=1, 边缘分布仍 N(0,1) — 写入值与被替换值同为白);
    随后带 k+1..K-1 换回原帧噪声 (FreeTraj 75% 高频重采样的频带版:
    k=3 ≈ 其 25% 共享能量档)。按构造保白性 → 不做方差重归一化
    (保「无刷新时框外逐位不变」; irfftn 数值漂移 ~1e-6 由容差覆盖)。
    k=K-1 时输出即全带 patch 写入 (框内跨帧恒等, 测试锚点)。
    """
    C, T, H, W = eps.shape
    g = torch.Generator().manual_seed(seed)
    full = eps.float().clone()
    for (y0, x0, hh, ww) in boxes:
        patch = torch.randn((C, 1, hh, ww), generator=g)
        full[:, :, y0:y0 + hh, x0:x0 + ww] = patch
    if k >= K - 1:
        return full.to(eps.dtype)                      # 全带写入旁路: 逐位精确
    keep = radial_band_masks_2d(H, W, K=K)
    mask = torch.stack([keep[b] for b in range(k + 1)]).sum(dim=0).bool()
    F_full = torch.fft.rfftn(full, dim=(-2, -1))
    F_eps = torch.fft.rfftn(eps.float(), dim=(-2, -1))
    out = torch.fft.irfftn(torch.where(mask, F_full, F_eps), s=(H, W), dim=(-2, -1))
    return out.to(eps.dtype)


def temporal_corr(x, box=None):
    """相邻帧 Pearson 相关均值 (patch 相干剂量度量)。box=(y0,x0,h,w) 时仅框内。

    白噪声跨帧独立 → ≈0; 跨帧共享 patch → 框内 ≈ 共享能量占比。
    """
    if box is not None:
        y0, x0, hh, ww = box
        v = x[:, :, y0:y0 + hh, x0:x0 + ww]
    else:
        v = x
    a = v[:, :-1].flatten(2).float()
    b = v[:, 1:].flatten(2).float()
    a = a - a.mean(dim=2, keepdim=True)
    b = b - b.mean(dim=2, keepdim=True)
    corr = (a * b).sum(dim=2) / (a.norm(dim=2) * b.norm(dim=2) + 1e-12)
    return float(corr.mean())


# ==========================================
# V 谷构造器 (2026-09-10 冒烟发现后增补, 见下)
# 冒烟实测: 合成井被读作背景产能 (井区纹理能量 5~10×, 主体坐安静侧)
# → 反转: 目标区压低低频 = 最未承诺区 = 吸收井。
# ==========================================

def valley_inject(eps, valleys, k=1, K=8):
    """谷构造器 · 吸收井: out = (eps − E·low_{≤k}(eps)) / 逐帧方差。

    valleys=[(y0,x0,r,depth),...]: 目标区低频成分压低 (1−depth·exp(−d²/2r²));
    白噪声谱平 vs 自然图像 1/f → 低频压低 = 更未承诺, 主体据冒烟证据坐安静侧。
    与 well_inject 同族 (线性作用, 联合高斯性保持, 逐帧方差重归一化)。
    depth 全零 → 逐位恒等。
    """
    if all(v[3] == 0 for v in valleys):
        return eps.clone()
    _, _, H, W = eps.shape
    E = well_envelope(H, W, valleys)
    low = low_pass_2d(eps, k, K)
    out = eps.float() - E[None, None] * low
    var = out.var(dim=(-2, -1), keepdim=True)
    out = out / (var + 1e-8).sqrt()
    return out.to(eps.dtype)


def smooth_well(eps, wells, k_keep=1, K=8):
    """压高频构造器 (smooth-well) · 载体解缠判别实验 (2026-09-11 增补)。

    out = ((1−E)·eps + E·low_{≤k_keep}(eps)) / 逐帧方差 — 区域内向自身低通混合:
      - 谱倾斜更 1/f 化 = 与井同号 (承诺度理论预言同样排斥);
      - 但区域**绝对低频能量不变** (低通分量原样通过, 高带 (1−E) 缩放) —
        区分「谱倾斜(相对)」vs「绝对低频能量」哪个是机制变量;
      - 区域空间变平滑 = 「已部分去噪」态, 可能长出 c2f 式模糊幽灵 (第三种区域命运)。
    剂量警告: 高带占白噪声能量 ~94%, depth 默认 0.5 (全深会出分布)。
    """
    if all(w[3] == 0 for w in wells):
        return eps.clone()
    _, _, H, W = eps.shape
    E = well_envelope(H, W, wells)
    low = low_pass_2d(eps, k_keep, K)
    out = (1.0 - E[None, None]) * eps.float() + E[None, None] * low
    var = out.var(dim=(-2, -1), keepdim=True)
    out = out / (var + 1e-8).sqrt()
    return out.to(eps.dtype)


def boost_high(eps, wells, k_keep=1, K=8):
    """抬高频构造器 (boost-high) · 载体解缠判别实验的第四臂 (2026-09-11 增补)。

    out = (eps + E·high_{>k_keep}(eps)) / 逐帧方差 = ((1+E)·eps − E·low) / var。
      - 区域谱倾斜更白化 = 与谷同号 (谱倾斜论预言吸引);
      - 绝对低频能量不变 (与 smooth 同一判别性质, 方向相反);
      - 局部总能量升高 (忙) = 忙度论预言排斥 — **谱倾斜论与忙度论唯一预测相反的臂**。
    剂量警告: 同 smooth, depth 默认 0.5 (高带能量占 ~94%)。
    """
    if all(w[3] == 0 for w in wells):
        return eps.clone()
    _, _, H, W = eps.shape
    E = well_envelope(H, W, wells)
    low = low_pass_2d(eps, k_keep, K)
    out = eps.float() + E[None, None] * (eps.float() - low)
    var = out.var(dim=(-2, -1), keepdim=True)
    out = out / (var + 1e-8).sqrt()
    return out.to(eps.dtype)


def well_inject_mask(eps, mask, amp=1.0, k=1, K=8):
    """轮廓模板井 (shape_v1, 2026-09-12): 掩码包络版 well_inject。

    out = (eps + amp·M·low_{≤k}(eps)) / 逐帧方差, M ∈ [0,1] 实体轮廓掩码
    (羽化, latent 分辨率 60×104)。与高斯 blob 的唯一差别是包络的形状语义
    (模板补全假设: 流形内形状被读作内容胚芽而非异常标记)。
    amp=0 → 逐位恒等; 逐帧方差重归一化与全家族同款。
    """
    if amp == 0:
        return eps.clone()
    low = low_pass_2d(eps, k, K)
    out = eps.float() + amp * mask.float()[None, None] * low
    var = out.var(dim=(-2, -1), keepdim=True)
    out = out / (var + 1e-8).sqrt()
    return out.to(eps.dtype)


def cn_splice(eps, c, gamma, k=1, K=8):
    """Colorful-Noise 式低频拼接 (SSN 算子, arXiv 2605.00548 配方, 2026-09-13)。

    ε 的 ≤k 径向带替换为 γ·c 的对应带, 其余带原样:
      out = F⁻¹( mask·γ·c_F + (1−mask)·ε_F )
    - 白性保持是**参数式**的 (CN 原设计): 带窄 (k 小) + 重衰减 (γ 小), 不做显式
      再白化; γ 是唯一剂量旋钮 (CN 工作区: sketch 0.04 / 自然图 0.2);
    - c = 布局目标图 (实体剪影 @ 目标位) 的 VAE 编码 —— 注入值是**真实图像低频
      的重衰减版** (流形内语义), 区别于包络井的幅度标记;
    - 精细轮廓不可携带 (中频=两侧实验判死); 可携带上限 = 剪影的 B0-k 体 plan。
    γ=0 → 逐位恒等。
    """
    if gamma == 0:
        return eps.clone()
    _, _, H, W = eps.shape
    Fe = torch.fft.rfftn(eps.float(), dim=(-2, -1))
    Fc = torch.fft.rfftn(c.float(), dim=(-2, -1))
    keep = radial_band_masks_2d(H, W, K=K)
    mask = torch.stack([keep[b] for b in range(k + 1)]).sum(dim=0).bool()
    F = torch.where(mask[None, None], gamma * Fc, Fe)
    return torch.fft.irfftn(F, s=(H, W), dim=(-2, -1)).to(eps.dtype)


def patch_transplant(eps, mask_src, dx, dy, k=1, K=8, feather=3.0):
    """实体低频补丁移植 (attract_v2 核心算子, 2026-09-13)。

    把 eps 自身低频场 (≤k 带) 中 mask_src 区域的**实际取值**逐帧搬移
    (dx,dy) 到目标位置, 羽化接缝融合:
      out_low = low·(1−M_t) + roll(low, (dy,dx))·M_t,  out = out_low + high。
    - 值全部来自同一真实场的重排 (无新值引入, 边缘高斯保持);
    - M_t = 掩码平移后的目标区羽化版 (接缝软融合);
    - 与 D1 的关系: D1 移植整条低频, 此处只搬「实体座位」补丁 ——
      若补丁足以吸引, 座位=可搬运; 若不足, 吸引需要整场语境。
    mask 全零/位移零 → 逐位恒等。
    """
    if mask_src.sum() == 0 or (dx == 0 and dy == 0):
        return eps.clone()
    _, _, H, W = eps.shape
    low = low_pass_2d(eps, k, K)
    high = eps.float() - low
    M_src = feather_mask(mask_src, feather)
    M_t = torch.roll(M_src, shifts=(dy, dx), dims=(-2, -1))
    M_t = (M_t / M_t.max().clamp(min=1e-9)).clamp(0, 1)
    low_shift = torch.roll(low, shifts=(dy, dx), dims=(-2, -1))
    out_low = low * (1 - M_t[None, None]) + low_shift * M_t[None, None]
    out = out_low + high
    var = out.var(dim=(-2, -1), keepdim=True)
    out = out / (var + 1e-8).sqrt()
    return out.to(eps.dtype)


def feather_mask(mask, feather=3.0):
    """二值/软掩码高斯羽化 (可分离核), 峰值归一。"""
    if feather <= 0:
        return mask.float()
    k = int(feather * 3) | 1
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    g1 = torch.exp(-ax ** 2 / (2 * feather ** 2))
    g1 = g1 / g1.sum()
    m = F.conv2d(F.conv2d(mask.float()[None, None], g1.view(1, 1, 1, -1),
                          padding=(0, k // 2)), g1.view(1, 1, -1, 1),
                 padding=(k // 2, 0))[0, 0]
    return m / m.max().clamp(min=1e-9)


def var_well(eps, wells, K=8):
    """方差井 (var-well) · 带特异性判别臂 (2026-09-12 增补)。

    out = (1+E)·eps / 逐帧方差 — 区域内全带统一幅度缩放:
      - Parseval 意义下 = 频率增益向量 g=(1+E, ..., 1+E) 的「纯方差、零剖面变化」点
        (τ 不变; well/valley/smooth/boost 是其余四向);
      - 空间白性精确保持: Cov(a_i·x_i, a_j·x_j)=0 (i≠j) — 全家族相关结构最干净;
      - 破坏仅剩: 异方差边缘 (区域内 N(0,s²)) + 时间相干包络;
      - T4 预言: ℓ×4 与 well 相同 → 排斥 ≈ well; 若显著弱 → 布局通道读的不是
        低带幅度本身; 代价预测: KL ≈ well ×16 (94% 系数承 λ=4), 崩/扛都是
        代价模型 ∝ 读取增益×偏离 的区域宽帽数据点。
    amp 全零 → 逐位恒等。
    """
    if all(w[3] == 0 for w in wells):
        return eps.clone()
    _, _, H, W = eps.shape
    E = well_envelope(H, W, wells)
    out = (1.0 + E[None, None]) * eps.float()
    var = out.var(dim=(-2, -1), keepdim=True)
    out = out / (var + 1e-8).sqrt()
    return out.to(eps.dtype)


# ==========================================
# S 天然井选择 (natsel, 2026-09-11): 不修改只选择
#   分布零破坏的位置控制候选 — 用白噪声自身的 χ² 涨落实现区域低带能量偏置。
#   选择的样本 = 条件分布 N(0,I)|A 的精确 draw (约束集内白律似然最高的代表),
#   区别于构造臂的多余相关结构。
# ==========================================

def region_band_energy(eps, y0, x0, r, k=1, K=8):
    """圆盘 (y0,x0,r) 内低带 (≤k) 分量的平均能量。选择扫描的度量。"""
    low = low_pass_2d(eps, k, K)
    C, T, H, W = low.shape
    yy = torch.arange(H).view(-1, 1)
    xx = torch.arange(W).view(1, -1)
    disk = ((yy - y0) ** 2 + (xx - x0) ** 2) <= r * r
    return float(low[..., disk].pow(2).mean())


def natural_pool_stats(eps_shape, seed_base, n_cand, y0, x_left, x_right, r,
                       sample_fn, k=1, K=8):
    """扫描 n_cand 个白噪声 draw → [(seed, E_left, E_right)] (只存标量, 不存张量)。

    sample_fn(shape, seed) 必须与生成协议逐位一致 (driver 传 sample_noise)。
    逐候选只做一次低通 (每帧一个 rFFT), n_cand=3000 全尺寸约 1 分钟 CPU。
    """
    C, T, H, W = eps_shape[1], eps_shape[2], eps_shape[3], eps_shape[4]
    yy = torch.arange(H).view(-1, 1)
    xx = torch.arange(W).view(1, -1)
    dL = ((yy - y0) ** 2 + (xx - x_left) ** 2) <= r * r
    dR = ((yy - y0) ** 2 + (xx - x_right) ** 2) <= r * r
    out = []
    for j in range(n_cand):
        seed = seed_base + j
        low = low_pass_2d(sample_fn(eps_shape, seed=seed)[0], k, K)
        out.append((seed, float(low[..., dL].pow(2).mean()),
                    float(low[..., dR].pow(2).mean())))
    return out
