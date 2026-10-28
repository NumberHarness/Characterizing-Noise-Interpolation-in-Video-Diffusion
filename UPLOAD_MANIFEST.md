# GitHub Upload Manifest

This manifest scopes the public repository to regenerating experiment videos,
YOLO measurements, and targeted sampling-step measurements. It does not include
paper analysis, plotting, manuscript files, or cached results.

## Include

- `.gitignore`
- `README.md`
- `UPLOAD_MANIFEST.md`
- `requirements-reproduction.txt`
- `model_config.py`
- `model_paths.env.example`
- `reproduce_experiments.py`
- `f0a_gen.py`
- `hh_followup_gen.py`
- `hh_followup_readout.py`
- `groups_gen.py`
- `ga2_gen.py`
- `x0_gen.py`
- `gridfill_gen.py`
- `t2v_gen.py`
- `pure5b_gen.py`
- `expand4_gen.py`
- `kresp_run.py`
- `korder_probe.py`
- `kframes_gen.py`
- `kframes_horse.py`
- `readout_arms.py`
- `yolo_detector.py`
- `video_protocol.py`
- `ent_readout.py`
- `arm_catalog.py`
- `ll_blend.py`
- `ot_common.py`
- `noise_struct.py`
- `prescan_pixel_hf.py`
- `payload/donor_cow_p_cond.png`
- `scenes/horse_f/cond.png`
- `scenes/boat_l/cond.png`

The `--sampling-visuals` option in `reproduce_experiments.py` runs
`kframes_gen.py` and `kframes_horse.py` to regenerate decoded intermediate
sampling-step frames for visual inspection.

The k-order probe uses public OWLv2 only as a fallback when YOLO misses a
decoded subject frame. Its checkpoint is downloaded by Transformers at
runtime; `transformers` is already included in the requirements.

The three conditioning images are required inputs. Their redistribution
licenses have not been verified in this workspace; confirm the rights before
publishing those files. If they cannot be redistributed, provide the same
images separately and retain the documented paths.

## Do Not Include

- Wan model directories, YOLO checkpoints, or any other model weights:
  `models/`, `yolo_weights/`, `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`
- Generated videos, temporary latents, and generated f0B/gray conditions:
  `*.mp4`, `*.mov`, `*.webm`, `*.npz`, `groups/f0B/*.png`,
  `x0/cond_gray.png`
- Runtime measurements, generated manifests, and cached results:
  `*.json` at any directory depth, including `readouts*.json`,
  `kresp_*.json`, `phase1_*.json`, and `*_manifest.json`
- Paper sources, figures, tables, compiled PDFs, and LaTeX build files:
  `paper/`
- Analysis, statistics, plotting, paper-consistency, and manuscript-build
  scripts, including `build_results.py`, `metrics3.py`, `plot_*.py`,
  `make_*.py`, `verify_*.py`, and `reproduce_analysis.sh`
- Exploratory generators, legacy pair-selection/readout scripts, experiment
  notes, generated plots, and tests not listed above
- Local credentials and environment files, including `.env*` and
  `model_paths.env`

## Entry Point

After installing `requirements-reproduction.txt`, configuring the Wan2.2
checkpoint, and confirming the conditioning-image rights, use:

```bash
python reproduce_experiments.py --dry-run
python reproduce_experiments.py
```

The experiment seed pairs are fixed in `arm_catalog.py`. YOLO is used to
recompute measurements from regenerated videos; cached measurements are not
inputs to this workflow.

## Validation Status

- The 13-stage driver dry-run completes, and the full test suite passed
  (157 tests).
- Four videos for a fixed HH pair (base, donor, LL@1, HH@1) were generated
  from the configured Wan2.2 checkpoint on an RTX 4080 SUPER. Both mixed arms
  passed the whiteness gate (standard deviations 1.00026 and 0.99977); YOLO
  measured all four on `cuda:0`, with 9/9 endpoint frames valid.
- Compared with the prior saved readouts for that seed pair, the regenerated
  videos' endpoint centers differed by at most 0.9 px. This is numerical
  agreement, not a claim of pixel-identical video generation.
- The full 11-stage video/readout workflow completed in an isolated copy
  initialized with the existing experiment videos. The generation stages
  resumed by skipping those existing videos; YOLO recomputed all 310 main
  matrix readouts (308 valid by the endpoint-frame rule, 2 invalid) and all
  28 HH follow-up readouts (28 valid), with no processing errors.
- Against the existing readout caches, all 310 main-matrix validity decisions
  matched; rounded positions differed by at most 0.1 px, with one video having
  8 rather than 9 detected endpoint frames (still valid). All 28 HH readouts
  matched exactly on the reported fields.
- This does **not** establish a from-empty full regeneration: the hundreds of
  main-matrix videos were not all regenerated in this run. Sampling-step
  probes are being checked separately; do not describe video regeneration
  from an empty directory as completed unless that run has been performed.
