"""ent_readout.py — 低剂量实体读出(M3 修订版, 2026-10-13)

读出协议(预注册于 ent_manifest.json):
  每 5 帧采样; 可信他框 = conf≥0.5 且与主框 IoU<0.2(上游判读层过滤);
  entity_tracks: 逐帧他框按邻近性(cx_tol, cy_tol, 允许 gap 个空帧)连成轨迹;
  new_entities: 轨迹需 (a) cy>cy_min —— 排除条件图继承的背景牛群带(cy≈23–60),
  (b) 位置不匹配本视频 f0 检出(f0_tol) —— 排除条件图内容, (c) 持续≥min_frames —— 排除毛刺。
纯函数, TDD 见 tests/test_ent_readout.py。
"""


def entity_tracks(others_by_frame, cx_tol=80, cy_tol=45, gap=1):
    """others_by_frame: {frame: [(cx, cy, w, conf), ...]} → 轨迹列表
    {frames, positions, confs, n, cx_mean, cy_mean}。贪心关联: 与轨迹末点邻近即延续;
    gap 按**采样序**计(允许 gap 个空采样, 与采样步长无关)。"""
    order = {f: i for i, f in enumerate(sorted(others_by_frame))}
    tracks = []
    for f in sorted(others_by_frame):
        for (cx, cy, w, conf) in others_by_frame[f]:
            best, bi = None, None
            for i, t in enumerate(tracks):
                if order[f] - t["last_rank"] > gap + 1:
                    continue
                lx, ly = t["positions"][-1]
                if abs(cx - lx) <= cx_tol and abs(cy - ly) <= cy_tol:
                    d = abs(cx - lx) + abs(cy - ly)
                    if best is None or d < best:
                        best, bi = d, i
            if bi is None:
                tracks.append({"frames": [f], "positions": [(cx, cy)],
                               "confs": [conf], "ws": [w], "last_rank": order[f]})
            else:
                t = tracks[bi]
                t["frames"].append(f)
                t["positions"].append((cx, cy))
                t["confs"].append(conf)
                t["ws"].append(w)
                t["last_rank"] = order[f]
    out = []
    for t in tracks:
        xs = [p[0] for p in t["positions"]]
        ys = [p[1] for p in t["positions"]]
        out.append({"frames": t["frames"], "n": len(t["frames"]),
                    "cx_mean": sum(xs) / len(xs), "cy_mean": sum(ys) / len(ys),
                    "conf_max": max(t["confs"]), "w_mean": sum(t["ws"]) / len(t["ws"])})
    return out


def new_entities(tracks, f0_others, cy_min=100, min_frames=2, f0_tol=40):
    """过滤出新实体: cy>cy_min 且不匹配 f0 位置且持续≥min_frames 的轨迹。"""
    out = []
    for t in tracks:
        if t["cy_mean"] <= cy_min or t["n"] < min_frames:
            continue
        if any(abs(t["cx_mean"] - fx) <= f0_tol and abs(t["cy_mean"] - fy) <= 60
               for fx, fy in f0_others):
            continue
        out.append(t)
    return out
