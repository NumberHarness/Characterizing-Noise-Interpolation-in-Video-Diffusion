"""Write per-video YOLO measurements for the fixed HH endpoint follow-up."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hh_followup_gen import PAIRS, arm_path, control_path
from readout_arms import readout_video

OUTPUT = ROOT / "hh_followup_readouts.json"


def video_entries():
    entries = []
    for scene, base, donor in PAIRS:
        pair = f"{scene} {base}-{donor}"
        for role, path in (
            ("base", control_path(scene, base)),
            ("donor", control_path(scene, donor)),
            ("ll@1", arm_path(scene, base, donor, "ll")),
            ("hh@1", arm_path(scene, base, donor, "hh")),
        ):
            entries.append((pair, role, scene, path))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="recompute all video measurements")
    parser.add_argument("--start-index", type=int, default=0,
                        help="zero-based index into the fixed video list")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output_path = args.output.expanduser().resolve()

    entries = video_entries()
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    entries = entries[args.start_index:]
    if args.max_videos is not None:
        entries = entries[:args.max_videos]

    missing = [path for _, _, _, path in entries if not path.is_file()]
    if missing:
        for path in missing:
            print(f"[missing] {path.relative_to(ROOT)}", flush=True)
        raise SystemExit("Generate the HH follow-up videos before measuring them.")

    data = {"meta": {
        "protocol": "YOLO11x, shared v4 per-video readout; no pairwise statistics",
        "updated": None,
    }, "videos": {}}
    if output_path.is_file() and not args.force:
        try:
            cached = json.loads(output_path.read_text())
            if cached.get("meta", {}).get("protocol") == data["meta"]["protocol"]:
                data = cached
        except (OSError, json.JSONDecodeError):
            pass

    for index, (pair, role, scene, path) in enumerate(entries, args.start_index + 1):
        key = f"{pair}|{role}"
        stat = path.stat()
        old = data["videos"].get(key)
        if (not args.force and old is not None
                and old.get("size") == stat.st_size
                and old.get("mtime") == round(stat.st_mtime, 3)):
            continue
        record = readout_video(path, scene, relaxed_boat=False)
        record.update({
            "pair": pair,
            "role": role,
            "path": str(path.relative_to(ROOT)),
            "size": stat.st_size,
            "mtime": round(stat.st_mtime, 3),
        })
        data["videos"][key] = record
        print(f"[{index}/{args.start_index + len(entries)}] {key} "
              f"f80={record.get('f80_cx')} valid={record.get('valid')}",
              flush=True)
        data["meta"]["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=1))

    data["meta"]["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"Wrote per-video measurements to {output_path.name}", flush=True)


if __name__ == "__main__":
    main()
