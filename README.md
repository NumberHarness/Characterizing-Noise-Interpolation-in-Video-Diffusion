# Characterizing Noise Interpolation in Video Diffusion

This repository contains the code needed to regenerate the paper's Wan2.2
experiment videos and YOLO measurement outputs. Generated videos, detector
readouts, sampling-step measurements, and model weights are runtime artifacts;
they are not required inputs and are not distributed in the repository.
Paper analysis, plotting, and LaTeX build workflows are outside this code
release. See [`UPLOAD_MANIFEST.md`](UPLOAD_MANIFEST.md) for the proposed
GitHub include/exclude list.

## Requirements

- Python 3.12
- A CUDA-capable GPU with enough memory for Wan2.2-TI2V-5B at 480 x 832 and
  81 frames
- FFmpeg (the Python `imageio-ffmpeg` package supplies a bundled executable)
- The public Diffusers-format Wan2.2-TI2V-5B checkpoint

Create an environment and install the generation/readout dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-reproduction.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The thread settings avoid invalid OpenMP defaults in some container
environments; they do not change the experiment's sampling parameters.

## Model Setup

Accept the Wan model terms on Hugging Face, authenticate if required, and
download the public Diffusers checkpoint:

```bash
hf auth login
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --revision b8fff7315c768468a5333511427288870b2e9635 \
  --local-dir models/Wan2.2-TI2V-5B-Diffusers
```

Set the local checkpoint path. The model files are not included:

```bash
export WAN22_MODEL_PATH="$PWD/models/Wan2.2-TI2V-5B-Diffusers"
python model_config.py --check wan22
```

YOLO11x is downloaded by Ultralytics on the first readout run. The checkpoint
used for the GPU validation had SHA-256
`7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24`.
To use a separately obtained checkpoint instead, set `YOLO11_WEIGHTS_PATH` to
its file:

```bash
export YOLO11_WEIGHTS_PATH="/path/to/yolo11x.pt"
```

The decoded-position k-order probe also uses the public
`google/owlv2-large-patch14-ensemble` checkpoint as a fallback only when YOLO
does not detect the subject. The fallback is pinned to revision
`95e26936e865f87db1742128404b3c035d47d89d`; Transformers downloads that
revision on the first fallback call.

## Reproduce

Preview the run order without loading models or creating outputs:

```bash
python reproduce_experiments.py --dry-run
```

Regenerate the experiment videos, YOLO readouts, and targeted sampling-step
measurements:

```bash
python reproduce_experiments.py
```

The driver uses the fixed seed pairs registered in `arm_catalog.py`; YOLO
recomputes measurements from generated videos and does not select or change
the experimental pairs. The generation settings are 81 frames at 480 x 832,
50 inference steps, guidance scale 5.0, and 16 fps.

Stages run in dependency order:

1. `f0a_gen.py`: f0A controls and all registered f0A interpolation arms.
2. `hh_followup_gen.py`: fixed controls and LL/HH endpoint arms for the seven
   non-dog pairs reported in the endpoint follow-up.
3. `groups_gen.py`: f0B conditioning images, donor controls, and LP comparison
   arms.
4. `ga2_gen.py`, `x0_gen.py`, `gridfill_gen.py`, and `t2v_gen.py`: f0B and
   complete gray-image sweeps.
5. `pure5b_gen.py` and `expand4_gen.py`: pure-T2V and remaining pair/regime
   combinations.
6. `readout_arms.py --force`: YOLO-based measurements for the registered
   4-regime, 4-pair experiment matrix (304 control/arm videos plus 6
   separately measured field@1 anchors).
7. `hh_followup_readout.py --force`: per-video YOLO readouts for the seven-pair
   HH endpoint follow-up. It writes raw video measurements only; it does not
   calculate cross-pair statistics.
8. `kresp_run.py all --force`: targeted latent sampling-step probes.
9. `korder_probe.py`: decoded-position sampling-step measurements using YOLO,
   with the OWL-ViT fallback described above.

To omit both sampling probes, use
`python reproduce_experiments.py --skip-sampling-probes`. Add
`--sampling-visuals` to also regenerate the decoded intermediate-frame images
used to inspect the cow and horse probes.

Each generator can be rerun independently. Non-empty generated videos are
skipped to support resuming an interrupted run. Move or remove local outputs
explicitly before regenerating them. `readout_arms.py --force` and
`kresp_run.py all --force` recompute their measurement outputs.

## Inputs and Outputs

The fixed experiment design is in `arm_catalog.py`. Required f0A conditioning
images are `payload/donor_cow_p_cond.png`, `scenes/horse_f/cond.png`, and
`scenes/boat_l/cond.png`; f0B images are extracted from newly generated
controls. The gray condition image is generated locally. The repository does
not include generated videos or measurement JSON files.

Runtime outputs are written beneath `videos/`, `lambda_scan/`, `scenes/`,
`groups/`, `ga2/`, `x0/`, `t2v/`, and `pure5b/`; YOLO measurements are written
to `readouts.json`, and sampling-step measurements to `kresp_*.json` and
`korder_probe.json`. The HH endpoint follow-up readouts are written to
`hh_followup_readouts.json`. These files are local outputs and should not be
committed.

Before redistributing the conditioning images, verify that their licenses
permit redistribution. If an image must be supplied separately, place it at
the corresponding path above before running the generation scripts.

## Reproduction Code

The generation and readout path uses:

- `reproduce_experiments.py`, `f0a_gen.py`, `groups_gen.py`, `ga2_gen.py`,
  `x0_gen.py`, `gridfill_gen.py`, `t2v_gen.py`, `pure5b_gen.py`, and
  `expand4_gen.py` for the main videos; `hh_followup_gen.py` regenerates the
  fixed seven-pair HH endpoint follow-up.
- `readout_arms.py`, `hh_followup_readout.py`, `yolo_detector.py`,
  `video_protocol.py`, and `ent_readout.py` for per-video measurements.
- `kresp_run.py` and `korder_probe.py` for latent and decoded-position
  sampling-step measurements. The k-order probe uses OWL-ViT only when YOLO
  misses a decoded subject frame.
- `arm_catalog.py`, `ll_blend.py`, `ot_common.py`, `noise_struct.py`,
  `prescan_pixel_hf.py`, and `model_config.py` for the experiment design,
  noise construction, whiteness checks, and model configuration. SciPy is
  included in `requirements-reproduction.txt` for the whiteness test.

`requirements-reproduction.txt` is the runtime dependency list. Analysis,
plotting, tests, paper sources, cached JSON measurements, and generated media
are not inputs to the reproduction command.

## Citation and Model Licenses

Please cite *Characterizing Noise Interpolation in Video Diffusion* when using
the experiment code or results. Wan checkpoints and YOLO weights are governed
by their respective upstream licenses and are not included here.
