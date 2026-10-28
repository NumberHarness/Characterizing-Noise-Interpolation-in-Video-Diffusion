"""readout_arms.py — YOLO-based video measurements, resumable by video

对 arm_catalog.video_entries() 的每条视频：
  端点: 帧 0,10,...,80（9 帧）→ f80_cx / med3_cx / n / valid（≥6/9）
  实体: 帧 0,5,...,80（17 帧）→ n_entities（cy>100, conf≥0.5, IoU<0.2, ≥2 采样帧）
同一次 YOLO 前向覆盖两种协议（5 步帧是 10 步帧的超集）。

判读口径:
  cow/horse: 最大面积主框（见 video_protocol.py）
  boat I2V:  宽度 150–260 的最高置信框
  boat Gray/Pure: 最高置信框（无宽度签名, 见论文 §4.2）

输出 readouts.json:
  {"meta": {...}, "videos": {key: {path, mtime, size, ...判读字段...}}}

用法:
  python readout_arms.py                 # 全量（约 310 条, GPU ~20 min）
  python readout_arms.py --max-videos 10 # 冒烟
  python readout_arms.py --only f0A|cow1 # 子集
  python readout_arms.py --force         # 全部重判
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arm_catalog as C

CACHE = HERE / "readouts.json"
CACHE_VERSION = 2          # 变更判读逻辑时递增, 旧缓存整体作废
READOUT_FRAMES = tuple(range(0, 81, 5))
ENDPOINT_FRAMES = tuple(range(0, 81, 10))


def select_main(bs, scene, relaxed_boat=False):
    """Apply the shared subject rule; gray/pure boats use max confidence."""
    from video_protocol import main_box
    if not bs:
        return None
    if relaxed_boat and scene == "boat_l":
        return max(bs, key=lambda t: t[4])
    return main_box(bs, scene)


def yolo_boxes(arr, scene: str):
    """Use YOLO only; a missing detection is counted as a missing frame.

    若某帧 YOLO 无检出，按协议计为缺失帧（≥6/9 门）。
    """
    from yolo_detector import detect_boxes
    from video_protocol import CLASS_IDS

    return detect_boxes(arr, CLASS_IDS[scene], conf=0.15, imgsz=1280)


def yolo_boxes_batch(frames, scene: str):
    """Use YOLO on sampled frames in batches with the same class/confidence rule."""
    from yolo_detector import detect_boxes_batch
    from video_protocol import CLASS_IDS

    return detect_boxes_batch(
        frames, CLASS_IDS[scene], conf=0.15, imgsz=1280, batch_size=8
    )


def readout_video(path: Path, scene: str, relaxed_boat: bool = False) -> dict:
    """一条视频的端点 + 实体判读。纯函数式（除 YOLO 外无状态）。"""
    import imageio.v3 as iio
    from ent_readout import entity_tracks, new_entities
    from video_protocol import iou_box

    per, others = {}, {}
    f0_others = []
    target_frames = set(READOUT_FRAMES)
    last_target = max(target_frames)
    sampled = []
    for f, frame in enumerate(iio.imiter(str(path))):
        if f > last_target:
            break
        if f not in target_frames:
            continue
        sampled.append((f, np.asarray(frame)[..., :3]))

    if not sampled or sampled[-1][0] != last_target:
        raise IndexError(f"Video does not contain frame {last_target}: {path}")

    boxes_by_frame = yolo_boxes_batch([arr for _, arr in sampled], scene)
    if len(boxes_by_frame) != len(sampled):
        raise RuntimeError("YOLO batch output count does not match sampled frames")

    for (f, _), bs in zip(sampled, boxes_by_frame):
        mb = select_main(bs, scene, relaxed_boat)
        if mb is None:
            continue
        if f in ENDPOINT_FRAMES:
            per[f] = {"cx": (mb[0] + mb[2]) / 2, "conf": mb[4],
                      "w": mb[2] - mb[0]}
        keep = [b for b in bs if b is not mb and b[4] >= 0.5
                and iou_box(mb, b) < 0.2]
        if keep:
            others[f] = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2,
                          b[2] - b[0], b[4]) for b in keep]
        if f == 0:
            f0_others = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in keep]

    ents = new_entities(entity_tracks(others), f0_others) if others else []
    rec = {
        "valid": False, "n": 0, "f80_cx": None, "med3_cx": None,
        "last_frame": None, "mean_conf": None, "cx_series": {},
        "n_entities": len(ents),
        "entities": [{"frames": e["frames"], "cx": round(e["cx_mean"]),
                      "cy": round(e["cy_mean"]), "conf": round(e["conf_max"], 2)}
                     for e in ents],
        "n_other_frames": len(others),
        "relaxed_boat": bool(relaxed_boat),
    }
    n = len(per)
    if n == 0:
        return rec
    last = max(per)
    seq = sorted(per)
    rec.update({
        "valid": n >= 6,
        "n": n,
        "f80_cx": round(per[last]["cx"], 1),
        "med3_cx": round(float(np.median([per[f]["cx"] for f in seq[-3:]])), 1),
        "last_frame": last,
        "mean_conf": round(float(np.mean([v["conf"] for v in per.values()])), 2),
        "cx_series": {str(f): round(per[f]["cx"], 1) for f in seq},
    })
    return rec


def load_cache(path: Path = CACHE) -> dict:
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("meta", {}).get("cache_version") == CACHE_VERSION:
            return data
        print(f"[cache] 版本不匹配, 旧 {path.name} 作废", flush=True)
    return {"meta": {}, "videos": {}}


def save_cache(data: dict, path: Path = CACHE):
    data["meta"].update({
        "cache_version": CACHE_VERSION,
        "protocol": "v4: endpoint 9 frames (>=6), dual readout, entity readout 17 frames; YOLO only",
        "entity_rule": "cy>100, conf>=0.5, IoU<0.2 vs main, not in f0, >=2 samples",
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_videos": len(data["videos"]),
    })
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="key 子串过滤")
    ap.add_argument("--start-index", type=int, default=0,
                    help="开始处理过滤后条目的零基索引")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--output", type=Path, default=CACHE,
                    help="measurement JSON output path")
    args = ap.parse_args()
    output_path = args.output.expanduser().resolve()

    entries = C.video_entries()
    if args.only:
        entries = [e for e in entries if args.only in e[0]]
    if args.start_index < 0:
        ap.error("--start-index must be non-negative")
    if args.start_index:
        entries = entries[args.start_index:]
    if args.max_videos:
        entries = entries[:args.max_videos]

    data = load_cache(output_path)
    done = skipped = failed = 0
    t0 = time.time()
    for i, (key, path, kind, regime, pair, mode, lam, scene, relaxed) in enumerate(entries, 1):
        if not path.exists():
            print(f"[missing] {key} {path}", flush=True)
            failed += 1
            continue
        st = path.stat()
        old = data["videos"].get(key)
        fresh = (old is not None and old.get("size") == st.st_size
                 and old.get("mtime") == round(st.st_mtime, 3)
                 and not args.force)
        if fresh:
            skipped += 1
            continue
        t = time.time()
        try:
            rec = readout_video(path, scene, relaxed)
        except Exception as e:                       # 解码/检测失败不吞掉整批
            rec = {"valid": False, "n": 0, "error": f"{type(e).__name__}: {e}",
                   "n_entities": None}
            print(f"[error] {key}: {rec['error']}", flush=True)
        rec.update({"path": str(path.relative_to(HERE)), "size": st.st_size,
                    "mtime": round(st.st_mtime, 3), "kind": kind,
                    "regime": regime, "pair": pair, "mode": mode, "lam": lam,
                    "scene": scene})
        data["videos"][key] = rec
        done += 1
        print(f"[{i}/{len(entries)}] {key} f80={rec.get('f80_cx')} "
              f"med3={rec.get('med3_cx')} ent={rec.get('n_entities')} "
              f"valid={rec.get('valid')} ({time.time()-t:.1f}s)", flush=True)
        if done % args.save_every == 0:
            save_cache(data, output_path)
    save_cache(data, output_path)
    print(f"\n→ {output_path.name}: judged {done}, skipped {skipped}, failed {failed}, "
          f"total {len(data['videos'])} in {time.time()-t0:.0f}s", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
