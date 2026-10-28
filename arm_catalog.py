"""arm_catalog.py — 论文实验矩阵的唯一路径注册表（2026-10-18）

把 4 conditioning regime × 4 seed pair × 3 mode × 6 λ 的每一条视频、每一个
控制视频、每一张条件图映射为唯一 key。readout_arms.py / build_results.py /
绘图脚本都只从这里取路径，避免任何“某个数字来自哪条视频”的隐式知识。

regime:
  f0A  WanImageToVideo, 场景原图 (scene photo)
  f0B  WanImageToVideo, 另一帧 (groups/f0B/<scene>.png)
  gray WanImageToVideo, 纯灰图 (x0/cond_gray.png)
  pure WanPipeline, 无图, 帧 0 参与混合
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

LAMS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
MODES = ("field", "ll", "lp")
REGIMES = ("f0A", "f0B", "gray", "pure")
REGIME_LABEL = {
    "f0A": "f0_A (informative)",
    "f0B": "f0_B (different image)",
    "gray": "Gray (zero info)",
    "pure": "Pure T2V (no image)",
}

# pair key -> scene / base seed / donor seed / paper label
PAIRS = {
    "cow1": {"scene": "cow_p", "sb": 9220, "sd": 9240, "label": "cow 9220-9240"},
    "horse": {"scene": "horse_f", "sb": 9300, "sd": 9240, "label": "horse 9300-9240"},
    "cow2": {"scene": "cow_p", "sb": 9200, "sd": 9250, "label": "cow 9200-9250"},
    "boat": {"scene": "boat_l", "sb": 9250, "sd": 9210, "label": "boat 9250-9210"},
}
PAIR_ORDER = ("cow1", "horse", "cow2", "boat")

# phase1 命名规则: 原始对不带 donor 后缀
ORIG = {"cow_p": (9200, 9250), "horse_f": (9200, 9260), "boat_l": (9250, 9210),
        "dog_p": (9220, 9200)}

SCENES = {"cow_p": "a cow standing in a green pasture",
          "horse_f": "a horse galloping across an open green field",
          "boat_l": "a sailboat sailing across a calm blue lake"}
PROMPTS = SCENES


def cell_key(regime: str, pair: str) -> str:
    p = PAIRS[pair]
    return f"{regime}|{p['scene']}|{p['sb']}\u2194{p['sd']}"


def parse_cell(key: str):
    regime, scene, seeds = key.split("|")
    sb, sd = (int(x) for x in seeds.split("\u2194"))
    return regime, scene, sb, sd


def _i2v_arm_name(mode: str, lam: float, scene: str, sb: int, sd: int) -> str:
    tag = f"i2v_lam_{mode}{int(round(lam * 100)):02d}_s{sb}"
    return tag + ("" if (sb, sd) == ORIG[scene] else f"d{sd}")


def _i2v_dir(regime: str, pair: str) -> Path:
    scene = PAIRS[pair]["scene"]
    if regime == "f0A":
        return HERE / ("lambda_scan" if scene == "cow_p" else f"scenes/{scene}")
    if regime == "f0B":
        return HERE / "ga2" / scene
    if regime == "gray":
        # cow 在 x0, horse/boat 在 t2v/<scene>
        return HERE / ("x0" if scene == "cow_p" else f"t2v/{scene}")
    raise KeyError(regime)


def _flat_arm_name(mode: str, lam: float, sb: int, sd: int) -> str:
    return f"{mode}{int(round(lam * 100)):02d}_s{sb}d{sd}"


def arm_path(regime: str, pair: str, mode: str, lam: float) -> Path:
    """λ 臂视频路径。

    f0A 存在两种历史命名：原始对不带 donor 后缀，扩展批带 d<sd> 后缀。
    两者数值等价（同条件图/同 base/donor seed/同算子）；优先返回实际存在的文件，
    否则返回规范命名（供 validate_paths 报缺口）。field@1.0 未单独生成时，
    readout/build 阶段用 donor 控制视频（锚恒等），不在这里替换。
    """
    assert mode in MODES and lam in LAMS
    p = PAIRS[pair]
    scene, sb, sd = p["scene"], p["sb"], p["sd"]
    if regime == "pure":
        return HERE / "pure5b" / scene / f"{_flat_arm_name(mode, lam, sb, sd)}.mp4"
    if regime == "f0A":
        d = _i2v_dir(regime, pair)
        candidates = [d / f"{_i2v_arm_name(mode, lam, scene, sb, sd)}.mp4",
                      d / f"i2v_lam_{_flat_arm_name(mode, lam, sb, sd)}.mp4"]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]
    return _i2v_dir(regime, pair) / f"{_flat_arm_name(mode, lam, sb, sd)}.mp4"


def field100_path(regime: str, pair: str) -> Path | None:
    """单独存在的 field λ=1 视频（用于锚点 MD5 复核）; 没有则 None。"""
    p = PAIRS[pair]
    scene, sb, sd = p["scene"], p["sb"], p["sd"]
    if regime == "f0A":
        return _i2v_dir(regime, pair) / f"{_i2v_arm_name('field', 1.0, scene, sb, sd)}.mp4"
    if regime == "f0B":
        return _i2v_dir(regime, pair) / f"{_flat_arm_name('field', 1.0, sb, sd)}.mp4"
    if regime == "pure":
        return HERE / "pure5b" / scene / f"{_flat_arm_name('field', 1.0, sb, sd)}.mp4"
    return None


def control_path(regime: str, pair: str, which: str) -> Path:
    """which ∈ {base, donor}。"""
    assert which in ("base", "donor")
    p = PAIRS[pair]
    scene = p["scene"]
    seed = p["sb"] if which == "base" else p["sd"]
    if regime == "f0A":
        if scene == "cow_p":
            return HERE / "videos" / f"cow_p_s{seed}_ctrlB.mp4"
        return HERE / "scenes" / scene / f"{scene}_s{seed}_ctrlB.mp4"
    if regime == "f0B":
        if which == "base":
            return HERE / "ga2" / scene / f"ctrlB_f0B_s{seed}.mp4"
        # donor ctrl 在 groups/ga（与 f0B 条件图 + donor seed）
        return HERE / "groups" / "ga" / scene / f"{scene}_s{seed}_ctrlB_f0B.mp4"
    if regime == "gray":
        d = HERE / ("x0" if scene == "cow_p" else f"t2v/{scene}")
        return d / f"ctrl_s{seed}.mp4"
    if regime == "pure":
        return HERE / "pure5b" / scene / f"ctrl_s{seed}.mp4"
    raise KeyError(regime)


def condition_image(regime: str, pair: str) -> Path | None:
    scene = PAIRS[pair]["scene"]
    if regime == "f0A":
        if scene == "cow_p":
            return HERE / "payload" / "donor_cow_p_cond.png"
        return HERE / "scenes" / scene / "cond.png"
    if regime == "f0B":
        return HERE / "groups" / "f0B" / f"{scene}.png"
    if regime == "gray":
        return HERE / "x0" / "cond_gray.png"
    return None


def readout_config(regime: str, pair: str) -> dict:
    """判读口径: boat 在 gray/pure 用 max-confidence（无宽度签名）。"""
    scene = PAIRS[pair]["scene"]
    return {"scene": scene,
            "relaxed_boat": scene == "boat_l" and regime in ("gray", "pure")}


def blend_frame0(regime: str) -> bool:
    """pure 用 WanPipeline, 全部 21 个 latent 帧参与混合; I2V 保留帧 0。"""
    return regime == "pure"


def cells():
    for regime in REGIMES:
        for pair in PAIR_ORDER:
            yield regime, pair


def arms():
    for regime, pair in cells():
        for mode in MODES:
            for lam in LAMS:
                yield regime, pair, mode, lam


def video_entries():
    """返回 [(key, path, kind, regime, pair, mode, lam, scene, relaxed_boat)]。

    kind ∈ {base, donor, arm, field100}. 只包含路径实际存在的条目;
    缺失路径由 validate_paths() 单独报告。
    """
    out = []
    for regime, pair in cells():
        cfg = readout_config(regime, pair)
        for which in ("base", "donor"):
            p = control_path(regime, pair, which)
            out.append((f"ctrl|{regime}|{pair}|{which}", p, which, regime, pair,
                        None, None, cfg["scene"], cfg["relaxed_boat"]))
        for mode in MODES:
            for lam in LAMS:
                if mode == "field" and lam == 1.0:
                    continue  # 端点用 donor 控制视频, 不重复判读
                p = arm_path(regime, pair, mode, lam)
                out.append((f"arm|{regime}|{pair}|{mode}|{lam}", p, "arm", regime,
                            pair, mode, lam, cfg["scene"], cfg["relaxed_boat"]))
        p100 = field100_path(regime, pair)
        if p100 is not None and p100.exists():
            out.append((f"field100|{regime}|{pair}", p100, "field100", regime, pair,
                        "field", 1.0, cfg["scene"], cfg["relaxed_boat"]))
    return out


def validate_paths():
    """返回 (present, missing) 两个列表; 每个元素是 (key, path)。"""
    present, missing = [], []
    for key, path, *_ in video_entries():
        (present if path.exists() else missing).append((key, path))
    return present, missing


def dof_table():
    """表 1: 每个 latent 帧的各模式实自由度。"""
    import numpy as np
    from ll_blend import lp_mask

    H, W = 30, 52
    m_lp, dof_lp = lp_mask(H, W, 0.25)
    ll_dof = (H // 2) * (W // 2)
    return {
        "field": {"dof": H * W, "fraction": 1.0},
        "ll": {"dof": ll_dof, "fraction": ll_dof / (H * W)},
        "lp": {"dof": int(dof_lp), "fraction": dof_lp / (H * W)},
        "hh": {"dof": H * W - ll_dof, "fraction": 1 - ll_dof / (H * W)},
    }
