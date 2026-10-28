"""ot_common.py — wan_object_trace 纯函数库(I2V 运动引导实验, 2026-09-27)

内容:
  几何:TI2V-5B latent 规格(48ch, 16×16×4 压缩)、I2V 可编辑帧、像素↔格换算、尾宽;
  算子:transplant_ll_block —— 把供体噪声在源块的 LL(level) 系数写入 base 噪声的
        目标块(仅可编辑帧 1..T-1),含 no-op 断言 / Parseval 剂量 / 白性门;
  指标:traj_centers / net_displacement / follow_score / paired_follow;
  判定:placebo_envelope / verdict_trace / ARMS_TRACE 注册表。

设计依据:`wan_object_trace/README.md`(预注册)。数学基础:`docs/wavelet_formulas.html`。
"""
import math

import numpy as np
import torch

# ---- TI2V-5B 几何(侦察实测 2026-09-27) ----
# 层号选择:30×52 网格上 db4 的 LL 基支撑 =(L-1)·2^level;level=3 → 56 > 30(边界效应重),
# 故默认 level=2(支撑 28 ≤ 30, LL 网格 8×13)。物理尺度:LL2 块 = 4×4 新格 = 64px
# = 旧 T2V 网格(60×104, 8px/格)上 LL3 的 8×8 格 = 64px —— 两线尺度对齐。
LAT_C = 48          # VAE z_dim = transformer in_channels
SCALE_S = 16        # VAE scale_factor_spatial
SCALE_T = 4         # VAE scale_factor_temporal
FILT_LEN = {"haar": 2, "db2": 4, "db4": 8, "sym4": 8}

# 移植用小波(2026-09-30 定): **Haar**。LL 基支撑尾宽 (L-1)·2^level 最小(Haar=2 格),
# 移植逐位封闭在目标块内;db4 尾宽 14 格 + periodization 环绕 ⇒ 扰动撒满全画面(色斑根因)。
WAVE_TRANSPLANT = "haar"


# ==========================================
# 几何
# ==========================================

def latent_shape_ot(frames=81, height=480, width=832):
    """→ (C,T,H,W);480×832/81 帧 ⇒ (48,21,30,52)。"""
    return (LAT_C, (frames - 1) // SCALE_T + 1,
            height // SCALE_S, width // SCALE_S)


def editable_frames(T):
    """I2V:首帧被条件图像占据并保持干净 ⇒ 可编辑噪声帧 = 1..T-1。"""
    return list(range(1, T))


def pix_to_cell(v, scale=SCALE_S):
    return int(round(v / scale))


def block_cells_from_pix(box_pix, scale=SCALE_S):
    x0, y0, x1, y1 = box_pix
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"空/反向像素框: {box_pix}")
    return (pix_to_cell(x0), pix_to_cell(y0), pix_to_cell(x1), pix_to_cell(y1))


def tail_cells(level, filt_len=FILT_LEN["db4"]):
    """第 level 层基函数的支撑尾宽(格) = (L-1)·2^level。"""
    return (filt_len - 1) * (2 ** level)


def site_delta_cells(box_cells, target_cell, W):
    """目标位(格坐标)→ latent 平移格数;越界时贴边保护。"""
    x0, y0, x1, y1 = box_cells
    cx = (x0 + x1) / 2
    dx = int(round(target_cell - cx))
    if x0 + dx < 0:
        dx = -x0
    if x1 + dx > W - 1:
        dx = W - 1 - x1
    return dx


# ==========================================
# 移植算子
# ==========================================

def _np(x, dtype=np.float64):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(dtype, copy=True)
    return np.asarray(x, dtype=dtype)


def transplant_ll_block(eps, donor, box_cells, dx_cells, dy_cells=0, level=2,
                        wave=WAVE_TRANSPLANT, mode="periodization", allow_noop=True,
                        guard_whiteness=False, src_frame=None, blend_beta=1.0,
                        feather_cells=0, edit_frames=None, schedule=None,
                        include_frame0=False, bands="ll"):
    """把 donor 在 box_cells 的 LL(level) 系数搬到 eps 的 box+dx 处。

    eps/donor: (C,T,H,W) torch 或 numpy;默认仅编辑帧 1..T-1(帧 0 = 条件图, 逐位不变)。
    返回 (out[torch.float32], meta)。meta 含 dose(Parseval 剂量)/ noop / frames /
    level / dx_cells / filt_len / box / dst_box。

    轨迹/突发扩展(2026-10-01 方案一+二):
      edit_frames: 可编辑帧子集(方案二突发窗, 如 [1,2,3]);None=全部可编辑帧。
      schedule:    {t: (dx_cells, dy_cells)} 逐帧写入位点(方案一轨迹斜坡);
                   未覆盖的帧回落到 dx_cells/dy_cells;帧 0 恒不被编辑。
      include_frame0: T2V 口径(无图像条件)使用 —— 老代码 wav_transplant 在 T2V 下写
                   全部帧(含帧 0), 而 T2V 的判读量恰是"帧 0 落座";I2V 禁用(帧 0 被条件图占据)。
      bands: "ll" = 只换低频带(现状, 改动=框内能量的 50%); "all" = 换 level-1 全部
                   四个子带(=老代码 full 模式整块拷贝, 改动=框内 200%, 整帧约 4×)。
                   实测(2026-10-01): ll → 可编辑帧总能量 17.4%; all → 69.4%。
    """
    import pywt
    e, d = _np(eps), _np(donor)
    assert e.ndim == 4 and e.shape == d.shape, (e.shape, d.shape)
    C, T, H, W = e.shape
    s = 2 ** level
    x0, y0, x1, y1 = box_cells
    ix0, iy0 = x0 // s, y0 // s
    ix1, iy1 = int(math.ceil(x1 / s)), int(math.ceil(y1 / s))
    dxs, dys = int(dx_cells) // s, int(dy_cells) // s

    # 系数空间边界断言(§11.6 no-op 教训:越界切片会静默吞掉移植)
    coeff_h = int(math.ceil(H / s))
    coeff_w = int(math.ceil(W / s))
    assert iy0 >= 0 and ix0 >= 0 and iy1 <= coeff_h and ix1 <= coeff_w, \
        f"源块 {box_cells} 越出 LL 系数网格 {(coeff_h, coeff_w)}"
    assert iy1 > iy0 and ix1 > ix0, f"空系数块: {box_cells} @ level={level}"

    assert bands in ("ll", "all"), bands
    assert bands == "ll" or level == 1, "bands='all' 目前仅实现 level=1(四子带同形)"
    frames = editable_frames(T)
    if include_frame0:               # T2V: 无图像条件帧, 全帧可编辑(老代码语义)
        frames = list(range(T))
    if edit_frames is not None:      # 方案二: 突发窗子集(帧 0/越界号必须拒绝)
        bad = [t for t in edit_frames if t not in frames]
        assert not bad, f"edit_frames 含不可编辑帧 {bad}(可编辑={frames[0]}..{frames[-1]})"
        frames = sorted(set(int(t) for t in edit_frames))

    out = e.copy()
    dose = 0.0
    n_edit = 0
    dst_sites = {}
    for c in range(C):
        for t in frames:
            if schedule is not None and t in schedule:   # 方案一: 逐帧位点
                dxs_t = int(schedule[t][0]) // s
                dys_t = int(schedule[t][1]) // s
            else:
                dxs_t, dys_t = dxs, dys
            jx0, jx1 = ix0 + dxs_t, ix1 + dxs_t
            jy0, jy1 = iy0 + dys_t, iy1 + dys_t
            assert jy0 >= 0 and jx0 >= 0 and jy1 <= coeff_h and jx1 <= coeff_w, \
                f"目标块越界: 帧{t} box+({dxs_t * s},{dys_t * s}) 系数区 " \
                f"({jy0},{jx0})-({jy1},{jx1}) > {(coeff_h, coeff_w)}"
            ce = pywt.wavedec2(e[c, t], wave, mode=mode, level=level)
            t_src = t if src_frame is None else int(src_frame)   # 接续: 从供体指定帧取块
            cd = pywt.wavedec2(d[c, t_src], wave, mode=mode, level=level)
            if bands == "all":   # 四子带(LL,LH,HL,HH)同索引区一起换 = 整块补丁替换
                for k in range(3):
                    blk_k = cd[1][k][iy0:iy1, ix0:ix1]
                    dose += float(((blk_k - ce[1][k][jy0:jy1, jx0:jx1]) ** 2).sum())
                    ce[1][k][jy0:jy1, jx0:jx1] = blk_k
            blk = cd[0][iy0:iy1, ix0:ix1]
            old = ce[0][jy0:jy1, jx0:jx1].copy()
            if blend_beta < 1.0:          # 能量配比: 保方差混合(注入 β 比例的供体内容)
                wb = math.sqrt(max(0.0, 1.0 - blend_beta))
                wd = math.sqrt(max(0.0, blend_beta))
                blk = wb * old + wd * blk
            if feather_cells > 0:         # 羽化: 块边界余弦过渡(消低频接缝)
                w = _feather_window(blk.shape, feather_cells)
                blk = old + w * (blk - old)
            dose += float(((blk - old) ** 2).sum())
            ce[0][jy0:jy1, jx0:jx1] = blk
            rec = pywt.waverec2(ce, wave, mode=mode)
            out[c, t] = rec[:H, :W]
            if c == 0:
                dst_sites[t] = (x0 + dxs_t * s, y0 + dys_t * s,
                                x1 + dxs_t * s, y1 + dys_t * s)
            n_edit += 1

    noop = dose <= 1e-12
    if noop and not allow_noop:
        raise AssertionError(
            f"移植是空操作(dose≈0): box={box_cells} dx={dx_cells} level={level} — "
            "检查层号/形状/模式/目标位置")
    if not noop:
        field_delta = float(((out[:, 1:] - e[:, 1:]) ** 2).sum())
        assert field_delta > 0, "重建后无差异 —— 移植未生效(§11.6 二次断言)"

    meta = {"dose": dose, "noop": bool(noop), "frames": frames, "level": level,
            "src_frame": src_frame, "blend_beta": float(blend_beta),
            "feather_cells": int(feather_cells),
            "dx_cells": int(dx_cells), "dy_cells": int(dy_cells),
            "filt_len": FILT_LEN.get(wave, 8),
            "wave": wave, "mode": mode, "box": tuple(box_cells),
            "dst_box": (x0 + int(dx_cells), y0 + int(dy_cells),
                        x1 + int(dx_cells), y1 + int(dy_cells)),
            "dst_sites": dst_sites,
            "edit_frames": list(frames),
            "include_frame0": bool(include_frame0), "bands": bands,
            "n_edited_tensors": n_edit}

    if guard_whiteness:
        g = whiteness_gate(out[:, 1:])
        meta["whiteness"] = g
        assert g["pass"], f"白性门未过: {g}"

    return torch.from_numpy(out.astype(np.float32)), meta


def dose_of(out, eps):
    """场差能量(仅可编辑帧)——应与 meta['dose'] 相等(Parseval)。"""
    o, e = _np(out, np.float32), _np(eps, np.float32)
    assert o.shape == e.shape
    return float(((o[:, 1:] - e[:, 1:]) ** 2).sum())


def whiteness_gate(x, std_tol=0.01, morans_tol=0.01):
    """统计合法性门:std≈1 / morans≈0 / ks 通过。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from noise_struct import ks_white, morans_i
    t = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
    t = t.float()
    if t.dim() == 3:
        t = t[None]
    std = float(t.std())
    mor = float(morans_i(t))
    ks = bool(ks_white(t))
    ok = (abs(std - 1.0) <= std_tol) and (abs(mor) <= morans_tol) and ks
    return {"pass": ok, "std": round(std, 5), "morans": round(mor, 5),
            "ks_white": ks}


# ==========================================
# 轨迹指标(轨迹 = [(cx, cy), ...],像素坐标,cy 向下为正)
# ==========================================

def traj_centers(boxes_per_frame):
    """[(cx,cy) 或 None] → 插值后的 [(cx,cy)];全缺失返回 None。"""
    idx = [i for i, b in enumerate(boxes_per_frame) if b is not None]
    if not idx:
        return None
    arr = np.array([boxes_per_frame[i] for i in idx], dtype=np.float64)
    out = []
    for i in range(len(boxes_per_frame)):
        if boxes_per_frame[i] is not None:
            j = idx.index(i)
            out.append((float(arr[j, 0]), float(arr[j, 1])))
        else:
            out.append((float(np.interp(i, idx, arr[:, 0])),
                        float(np.interp(i, idx, arr[:, 1]))))
    return out


def net_displacement(traj):
    """净位移向量 (dx, dy) = 末帧中心 − 首帧中心。"""
    if not traj:
        return (0.0, 0.0)
    return (traj[-1][0] - traj[0][0], traj[-1][1] - traj[0][1])


def _unit(v):
    n = math.hypot(v[0], v[1])
    return (0.0, 0.0) if n < 1e-12 else (v[0] / n, v[1] / n)


def follow_score(traj, axis):
    """净位移在给定方向上的余弦投影 ∈[-1,1];零位移 → 0.0。"""
    d = net_displacement(traj)
    if math.hypot(*d) < 1e-9:
        return 0.0
    u, a = _unit(d), _unit(axis)
    return float(max(-1.0, min(1.0, u[0] * a[0] + u[1] * a[1])))


def paired_follow(trajs_treat, trajs_ctrl, axes):
    """同 seed 配对差:follow(处理) − follow(对照)。"""
    return [follow_score(t, a) - follow_score(c, a)
            for t, c, a in zip(trajs_treat, trajs_ctrl, axes)]


# ==========================================
# 判定
# ==========================================

def placebo_envelope(vals):
    return max((abs(v) for v in vals), default=0.0)


def verdict_trace(deltas, envelope, quality_ok, k, placebo_delta=None):
    """预注册判定(README §五)。deltas = 各 seed 的 follow 配对差。"""
    if not quality_ok:
        return "无效(质量门未过), 不出阴性"
    if not deltas:
        return "无数据"
    med = float(np.median(deltas))
    same = sum(1 for x in deltas if x > 0)
    if placebo_delta:
        if all(abs(p) >= 0.8 * abs(med) for p in placebo_delta) and abs(med) > envelope * 0.5:
            return "H2: 通用写入偏置(安慰剂同量级)"
    if med > envelope and same >= max(1, k - 1):
        return "H1: 运动引导成立"
    return "H0: 零效应"


# ==========================================
# 臂注册表(README §三)
# ==========================================

# v2(2026-09-28 冒烟后重设计):供体实体在画面正中 ⇒ 镜像位无空间;
# 改为**定向平移位**(上/下, 幅度 8 格 = 128px),水平同理可扩展。
# v3(2026-09-29 用户裁定): 目标位 = 距离实体当前位置**最远的角**(2D, 位移最大)
ARMS_TRACE = {
    "ctrlB":  {"content": "none",  "corner": False},
    "wavC":   {"content": "donor", "corner": True},    # 供体座位块 → 最远角
    "freshC": {"content": "fresh", "corner": True},    # 新鲜块 → 同一最远角(安慰剂)
    "shift0": {"content": "donor", "corner": False},   # 供体块原位重写(剂量对照)
}
SEAT_SIZE_CELLS = 8           # 座位块固定尺寸: 8×8 格 = 128×128 px
SEAT_LEVEL = 1                # 小块实体用 level=1(每系数 2 格 = 32px; 16 个系数/块)


# ==========================================
# 位点选择(系数空间裁剪 + 符号保护, 2026-09-27 自检修正)
# ==========================================

def _coeff_block_idx(box_cells, s):
    x0, y0, x1, y1 = box_cells
    return (y0 // s, x0 // s, int(math.ceil(y1 / s)), int(math.ceil(x1 / s)))


def mirror_site_cells(box_cells, W, level=2):
    """镜像位:目标块中心 = 画面中轴镜像(格坐标)。

    dx 对齐 2^level(系数空间可整数平移);在**系数空间**裁剪保证块在界内;
    若裁剪把位移改成 0 或翻转符号 → feasible=False(该供体无此方向空间)。
    返回 dict(dx_cells, dx_requested, dst_box_cells, pixel_shift, feasible, reason)。
    """
    s = 2 ** level
    x0, y0, x1, y1 = box_cells
    cx = (x0 + x1) / 2
    dx_req = int(round(((W - 1) - cx - cx) / s)) * s          # 对齐 s
    iy0, ix0, iy1, ix1 = _coeff_block_idx(box_cells, s)
    coeff_w = int(math.ceil(W / s))
    w = ix1 - ix0
    jx0 = ix0 + dx_req // s
    jx0 = max(0, min(jx0, coeff_w - w))
    dx_eff = (jx0 - ix0) * s
    ok = (dx_eff != 0) and (dx_eff * dx_req > 0)
    return {"dx_cells": dx_eff, "dx_requested": dx_req,
            "dst_box_cells": (x0 + dx_eff, y0, x1 + dx_eff, y1),
            "pixel_shift": dx_eff * SCALE_S, "feasible": ok,
            "reason": "" if ok else ("位移被裁剪为 0" if dx_eff == 0 else "符号被翻转")}


def site_dx_cells(box_cells, side, frac=0.5, W=52, level=2):
    """定向位移(增益标定用):side ∈ {+1,-1},frac = 占可用余量的比例。"""
    s = 2 ** level
    x0, y0, x1, y1 = box_cells
    iy0, ix0, iy1, ix1 = _coeff_block_idx(box_cells, s)
    coeff_w = int(math.ceil(W / s))
    w = ix1 - ix0
    room = (coeff_w - w - ix0) if side > 0 else ix0
    d = int(round(frac * max(room, 0))) * side
    dx_eff = d * s
    return {"dx_cells": dx_eff, "pixel_shift": dx_eff * SCALE_S,
            "dst_box_cells": (x0 + dx_eff, y0, x1 + dx_eff, y1),
            "feasible": dx_eff != 0, "room_cells": room}


# ==========================================
# 座位块与定向位点(v2, 2026-09-28)
# ==========================================

def seat_block_cells(box_cells, size_cells=SEAT_SIZE_CELLS, level=SEAT_LEVEL,
                     H=None, W=None):
    """以实体框中心为中心的固定尺寸座位块(格), 对齐系数网格。
    小块实体(60~90px)在 level=2 下只剩 1~2 个系数 ⇒ 固定 8×8 格(128px)+level1
    才能让"图案"有足够自由度(16 个 LL1 系数)。"""
    s = 2 ** level
    assert size_cells % s == 0, f"块尺寸 {size_cells} 必须是系数尺度 {s} 的倍数"
    x0, y0, x1, y1 = box_cells
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0 = int(round((cx - size_cells / 2) / s)) * s
    by0 = int(round((cy - size_cells / 2) / s)) * s
    if W is not None:                      # 贴边裁剪: 块必须完全在画面内(保持网格对齐)
        bx0 = max(0, min(bx0, (W - size_cells) - (W - size_cells) % s))
    if H is not None:
        by0 = max(0, min(by0, (H - size_cells) - (H - size_cells) % s))
    return (bx0, by0, bx0 + size_cells, by0 + size_cells)


def offset_site_cells(box_cells, dy_cells, dx_cells, H, W, level=SEAT_LEVEL):
    """块整体平移 (dy,dx) 格; 系数空间裁剪; 回报实际位移与符号保持。"""
    s = 2 ** level
    x0, y0, x1, y1 = box_cells
    assert dy_cells % s == 0 and dx_cells % s == 0, "位移必须是系数尺度的倍数"
    iy0, ix0, iy1, ix1 = _coeff_block_idx(box_cells, s)
    ch, cw = int(math.ceil(H / s)), int(math.ceil(W / s))
    w, h = ix1 - ix0, iy1 - iy0
    jy0 = max(0, min(iy0 + dy_cells // s, ch - h))
    jx0 = max(0, min(ix0 + dx_cells // s, cw - w))
    dy_eff, dx_eff = (jy0 - iy0) * s, (jx0 - ix0) * s
    ok = ((dy_eff * dy_cells > 0) or (dy_cells == 0)) and \
         ((dx_eff * dx_cells > 0) or (dx_cells == 0)) and (dy_eff or dx_eff or
                                                           (dy_cells == dx_cells == 0))
    return {"dy_cells": dy_eff, "dx_cells": dx_eff,
            "dy_requested": dy_cells, "dx_requested": dx_cells,
            "dst_box_cells": (x0 + dx_eff, y0 + dy_eff, x1 + dx_eff, y1 + dy_eff),
            "pixel_shift": (dx_eff * SCALE_S, dy_eff * SCALE_S),
            "feasible": bool(ok)}


# ==========================================
# 有符号位移 + 归一化完成度
# (对齐 `实体搬运完成度_有符号位移_20260929.md`;口径逐条照搬)
# ==========================================
NEED_FLOOR = 60.0            # |need| < 此值 → 剔除(分母趋零会放大)
NEED_INDETERMINATE = 150.0   # |need| < 此值 → 标"无法判定"(抖动未标定前, 文档 §4.1)
JITTER_FACTOR = 4.0          # 效应需 > 4× 检测抖动才可信


def signed_shift(arm_pos, ctrl_pos, target_pos):
    """有符号位移与归一化完成度。

    delta      = arm − ctrl        有符号(正 = 朝 target 方向)
    need       = target − ctrl     落到目标中心所需位移
    completion = delta / need      1.0 = 正好落到中心; 0 = 未动; <0 = 反向
    """
    delta = float(arm_pos) - float(ctrl_pos)
    need = float(target_pos) - float(ctrl_pos)
    completion = float("nan") if abs(need) < 1e-9 else delta / need
    return {"delta": delta, "need": need, "completion": completion}


def gate_need(need):
    """按 |need| 给可判定性分级:剔除 / 无法判定 / 可用。"""
    a = abs(float(need))
    if a < NEED_FLOOR:
        return "剔除"
    if a < NEED_INDETERMINATE:
        return "无法判定"
    return "可用"


def jitter_range(values):
    """检测抖动 = 逐帧读数的极差(文档 §4.1 口径)。"""
    v = [float(x) for x in values if x is not None]
    return 0.0 if len(v) < 2 else max(v) - min(v)


def trust_effect(delta, jitter, factor=JITTER_FACTOR):
    """效应是否可信: |delta| > factor × jitter。"""
    return abs(float(delta)) > factor * max(float(jitter), 1e-9)


# ==========================================
# 位移规划(2026-09-29: 修 "need<150px 不过门")
# ==========================================
MAX_OFFSET_CELLS = 12          # 位移封顶 12 格 = 192px
MIN_USEFUL_SHIFT_PX = 160      # 过参考文档 150px 判别门所需的最小位移


def plan_sites(seat_cells, H, W, level=SEAT_LEVEL, max_cells=MAX_OFFSET_CELLS):
    """按座位位置规划上/下最大可行位移(系数空间裁剪 + 封顶), 并给出是否过门。

    返回 {"wavD": {...}, "wavU": {...}, "overall_ok": bool}
    每个方向: dy_cells(实际), pixel_shift, feasible, dst_box_cells。
    """
    s = 2 ** level
    iy0, ix0, iy1, ix1 = _coeff_block_idx(seat_cells, s)
    ch = int(math.ceil(H / s))
    h = iy1 - iy0
    room_dn = (ch - h - iy0) * s          # 向下可用格数
    room_up = iy0 * s                     # 向上可用格数
    cap = max_cells - (max_cells % s)
    out = {}
    for name, room, sign in (("wavD", room_dn, +1), ("wavU", room_up, -1)):
        mag = min(cap, room - (room % s))
        dy = sign * mag
        r = {"dy_cells": dy, "pixel_shift": (0, dy * SCALE_S),
             "feasible": dy != 0,
             "dst_box_cells": (seat_cells[0], seat_cells[1] + dy,
                               seat_cells[2], seat_cells[3] + dy),
             "room_cells": room}
        out[name] = r
    best = max(abs(out["wavD"]["pixel_shift"][1]), abs(out["wavU"]["pixel_shift"][1]))
    out["overall_ok"] = best >= MIN_USEFUL_SHIFT_PX
    return out


# ==========================================
# 最远角目标位(2026-09-29 用户裁定: "移动到距离当前位置最远的那个角")
# ==========================================

CORNERS = {"TL": (0, 0), "TR": (1, 0), "BL": (0, 1), "BR": (1, 1)}


def farthest_corner_site(seat_cells, H, W, level=SEAT_LEVEL):
    """把座位块(同尺寸)贴到**离座位中心最远**的角(flush), 返回 2D 位移与几何量。

    dst 块贴角对齐系数网格;dx/dy 为格数位移(两者可同时非零)。
    """
    s = 2 ** level
    x0, y0, x1, y1 = seat_cells
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    best = None
    for name, (fx, fy) in CORNERS.items():
        dx0 = 0 if fx == 0 else W - bw
        dy0 = 0 if fy == 0 else H - bh
        dx0 -= dx0 % s
        dy0 -= dy0 % s
        ccx, ccy = dx0 + bw / 2, dy0 + bh / 2
        dist = math.hypot(ccx - cx, ccy - cy)
        if best is None or dist > best["distance_cells"]:
            best = {"corner": name, "dst_box_cells": (dx0, dy0, dx0 + bw, dy0 + bh),
                    "site_center_cells": (ccx, ccy),
                    "dx_cells": int(dx0 - x0), "dy_cells": int(dy0 - y0),
                    "distance_cells": dist}
    r = dict(best)
    r["shift_px"] = (r["dx_cells"] * SCALE_S, r["dy_cells"] * SCALE_S)
    r["distance_px"] = float(np.hypot(*r["shift_px"]))
    r["feasible"] = bool(abs(r["dx_cells"]) + abs(r["dy_cells"]) > 0)
    return r


def signed_completion_2d(arm_pos, ctrl_pos, target_pos):
    """2D 版有符号完成度(与文档 1D 口径同构, 方向 = 基线位置 → 目标)。

    delta      = (arm − ctrl) · û        û = unit(target − ctrl)
    need       = |target − ctrl|
    completion = delta / need
    """
    ax, ay = float(arm_pos[0]) - float(ctrl_pos[0]), float(arm_pos[1]) - float(ctrl_pos[1])
    tx, ty = float(target_pos[0]) - float(ctrl_pos[0]), float(target_pos[1]) - float(ctrl_pos[1])
    need = math.hypot(tx, ty)
    if need < 1e-9:
        return {"delta": 0.0, "need": 0.0, "completion": float("nan")}
    ux, uy = tx / need, ty / need
    delta = ax * ux + ay * uy
    return {"delta": delta, "need": need, "completion": delta / need}


def jitter_radial(traj, k=5):
    """2D 径向抖动: 轨迹对 k 帧滑动均值的残差范数的最大值。"""
    pts = [(float(p[0]), float(p[1])) for p in traj]
    n = len(pts)
    if n < 3:
        return 0.0
    res = []
    half = k // 2
    for i in range(half, n - half):          # 只取窗口完整的内部点(边界残差是窗口截断伪影)
        lo, hi = i - half, i + half + 1
        mx = float(np.mean([p[0] for p in pts[lo:hi]]))
        my = float(np.mean([p[1] for p in pts[lo:hi]]))
        res.append(math.hypot(pts[i][0] - mx, pts[i][1] - my))
    return max(res)


# ==========================================
# 续写忠实度(2026-09-30): 条件帧实体位置 → 生成视频 f1 的位置应连续
# ==========================================
FAITHFUL_MAX_JUMP_PX = 100.0


def continuation_jump(pos_cond, pos_f1):
    """条件帧实体位置与生成视频首帧之后位置的欧氏距离(px)。"""
    return float(math.hypot(float(pos_f1[0]) - float(pos_cond[0]),
                            float(pos_f1[1]) - float(pos_cond[1])))


def faithful_continuation(pos_cond, pos_f1, max_jump=FAITHFUL_MAX_JUMP_PX):
    """续写是否忠实: 实体在 f1 仍留在条件帧位置附近(默认 100px 内)。"""
    return continuation_jump(pos_cond, pos_f1) <= max_jump


def _feather_window(shape, cells):
    """块内余弦过渡窗(numpy): 最外 cells 格从 0 升到 1, 内部恒 1。"""
    h, w = int(shape[0]), int(shape[1])
    n = max(1, int(cells))
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))) if n >= 2 \
        else np.array([0.5])
    ky, kx = np.ones(h), np.ones(w)
    my, mx = min(n, h // 2), min(n, w // 2)
    if my > 0:
        ky[:my] = ramp[:my]; ky[-my:] = ramp[:my][::-1]
    if mx > 0:
        kx[:mx] = ramp[:mx]; kx[-mx:] = ramp[:mx][::-1]
    return ky[:, None] * kx[None, :]
