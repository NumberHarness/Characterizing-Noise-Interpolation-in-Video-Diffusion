"""Regenerate paper experiment videos and YOLO measurement outputs."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def stages(include_sampling_visuals: bool) -> list[tuple[str, list[str]]]:
    py = sys.executable
    result = [
        ("f0A controls and fixed-pair arms",
         [py, "f0a_gen.py"]),
        ("fixed seven-pair HH endpoint follow-up",
         [py, "hh_followup_gen.py"]),
        ("f0B images, donor controls, and LP comparison arms",
         [py, "groups_gen.py"]),
        ("f0B sweeps for cow1 and horse",
         [py, "ga2_gen.py"]),
        ("gray-image cow sweeps",
         [py, "x0_gen.py"]),
        ("remaining gray-image cow lambda points",
         [py, "gridfill_gen.py"]),
        ("gray-image horse sweep",
         [py, "t2v_gen.py"]),
        ("pure-T2V cow1 and horse sweeps",
         [py, "pure5b_gen.py"]),
        ("remaining cow2 and boat arms across all regimes",
         [py, "expand4_gen.py"]),
        ("YOLO readout for the registered experiment matrix",
         [py, "readout_arms.py", "--force"]),
        ("YOLO raw readout for the HH endpoint follow-up",
         [py, "hh_followup_readout.py", "--force"]),
        ("sampling-step latent probes",
         [py, "kresp_run.py", "all", "--force"]),
        ("decoded-position korder probe",
         [py, "korder_probe.py"]),
    ]
    if include_sampling_visuals:
        result.extend([
            ("decoded sampling-step frames for cow",
             [py, "kframes_gen.py"]),
            ("decoded sampling-step frames for horse",
             [py, "kframes_horse.py"]),
        ])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the stages without loading models or generating files",
    )
    parser.add_argument(
        "--skip-sampling-probes", "--skip-sampling-step",
        dest="skip_sampling_probes", action="store_true",
        help="skip latent and decoded-position sampling-step probes",
    )
    parser.add_argument(
        "--sampling-visuals", action="store_true",
        help="also regenerate decoded intermediate frames used in the paper",
    )
    args = parser.parse_args()

    if not args.dry_run:
        subprocess.run(
            [sys.executable, "model_config.py", "--check", "wan22"],
            cwd=ROOT,
            check=True,
        )

    run_stages = stages(args.sampling_visuals)
    if args.skip_sampling_probes:
        run_stages = [stage for stage in run_stages
                      if stage[1][1] not in ("kresp_run.py", "korder_probe.py")]

    for index, (label, command) in enumerate(run_stages, 1):
        print(f"\n[{index}/{len(run_stages)}] {label}", flush=True)
        print("$ " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    if args.dry_run:
        print("\nDry run complete; no models or output files were accessed.")
    else:
        print("\nExperiment video generation and readout complete.")


if __name__ == "__main__":
    main()
