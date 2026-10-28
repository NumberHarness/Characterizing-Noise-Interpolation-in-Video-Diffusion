"""Regenerate the paper's fixed seven-pair HH endpoint follow-up."""
from __future__ import annotations

import gc
import argparse
import time
from pathlib import Path

import torch

from ll_blend import blend_eps
from model_config import WAN22_MODEL_PATH
from ot_common import latent_shape_ot, whiteness_gate

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT
SHAPE = latent_shape_ot(81, 480, 832)
SCENES = {
    "cow_p": ("a cow standing in a green pasture",
              ROOT / "payload/donor_cow_p_cond.png"),
    "horse_f": ("a horse galloping across an open green field",
                ROOT / "scenes/horse_f/cond.png"),
    "boat_l": ("a sailboat sailing across a calm blue lake",
               ROOT / "scenes/boat_l/cond.png"),
}
ORIGINAL_PAIRS = {
    "cow_p": (9200, 9250),
    "horse_f": (9200, 9260),
    "boat_l": (9250, 9210),
}
PAIRS = (
    ("cow_p", 9200, 9250),
    ("cow_p", 9270, 9260),
    ("cow_p", 9220, 9240),
    ("horse_f", 9200, 9260),
    ("horse_f", 9300, 9240),
    ("boat_l", 9250, 9210),
    ("boat_l", 9280, 9290),
)


def sample_noise(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(SHAPE, generator=generator)


def output_dir(scene: str) -> Path:
    if scene == "cow_p":
        return OUTPUT_ROOT / "lambda_scan"
    return OUTPUT_ROOT / "scenes" / scene


def control_path(scene: str, seed: int) -> Path:
    if scene == "cow_p":
        return OUTPUT_ROOT / "videos" / f"cow_p_s{seed}_ctrlB.mp4"
    return OUTPUT_ROOT / "scenes" / scene / f"{scene}_s{seed}_ctrlB.mp4"


def arm_path(scene: str, base: int, donor: int, mode: str) -> Path:
    tag = f"i2v_lam_{mode}100_s{base}"
    if (base, donor) != ORIGINAL_PAIRS[scene]:
        tag += f"d{donor}"
    return output_dir(scene) / f"{tag}.mp4"


def jobs():
    """List the fixed control and band-replacement videos for all seven pairs."""
    entries = []
    for scene, base, donor in PAIRS:
        entries.extend(
            (scene, seed, "control", control_path(scene, seed))
            for seed in (base, donor)
        )
        entries.extend(
            (scene, (base, donor), mode, arm_path(scene, base, donor, mode))
            for mode in ("ll", "hh")
        )
    unique = {}
    for entry in entries:
        unique[entry[-1]] = entry
    return list(unique.values())


def generate_scene(scene: str, selected_paths: set[Path] | None = None) -> None:
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    from PIL import Image

    prompt, condition_path = SCENES[scene]
    if not condition_path.is_file():
        raise FileNotFoundError(f"Missing conditioning image: {condition_path}")
    image = Image.open(condition_path).convert("RGB")
    pipe = None
    try:
        for job_scene, seeds, kind, path in jobs():
            if job_scene != scene:
                continue
            if selected_paths is not None and path not in selected_paths:
                continue
            if path.is_file() and path.stat().st_size:
                print(f"[skip] {path}", flush=True)
                continue
            if pipe is None:
                pipe = WanImageToVideoPipeline.from_pretrained(
                    WAN22_MODEL_PATH,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                pipe.vae.enable_tiling()
                pipe.to("cuda")

            if kind == "control":
                eps = sample_noise(seeds).unsqueeze(0)
            else:
                base, donor = seeds
                eps = blend_eps(
                    sample_noise(base), sample_noise(donor), 1.0, band=kind
                ).unsqueeze(0)
                gate = whiteness_gate(eps[0, :, 1:])

            path.parent.mkdir(parents=True, exist_ok=True)
            start = time.time()
            output = pipe(
                image=image,
                prompt=prompt,
                height=480,
                width=832,
                num_frames=81,
                num_inference_steps=50,
                guidance_scale=5.0,
                latents=eps,
                output_type="np",
            )
            export_to_video(output.frames[0], str(path), fps=16)
            del output
            torch.cuda.empty_cache()
            gate_text = (f" whiteness={gate['pass']} std={gate['std']:.5f}"
                         if kind != "control" else "")
            print(f"[{path}]{gate_text} "
                  f"{time.time() - start:.0f}s", flush=True)
    finally:
        if pipe is not None:
            del pipe
            gc.collect()
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", action="append", default=[],
        help="limit generation to SCENE:BASE:DONOR; may be repeated",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT,
        help="write generated videos beneath this directory",
    )
    args = parser.parse_args()
    global OUTPUT_ROOT
    OUTPUT_ROOT = args.output_root.expanduser().resolve()

    selected: dict[str, set[Path]] | None = None
    if args.pair:
        selected = {}
        pair_lookup = {(scene, base, donor) for scene, base, donor in PAIRS}
        for value in args.pair:
            try:
                scene, base, donor = value.split(":")
                pair = (scene, int(base), int(donor))
            except ValueError as error:
                raise SystemExit(
                    f"Invalid --pair {value!r}; expected SCENE:BASE:DONOR"
                ) from error
            if pair not in pair_lookup:
                raise SystemExit(f"Pair is not registered: {value}")
            scene, base, donor = pair
            selected.setdefault(scene, set()).update((
                control_path(scene, base),
                control_path(scene, donor),
                arm_path(scene, base, donor, "ll"),
                arm_path(scene, base, donor, "hh"),
            ))

    for scene in (selected if selected is not None else SCENES):
        print(f"=== HH follow-up: {scene} ===", flush=True)
        generate_scene(scene, selected[scene] if selected is not None else None)
    print("HH_FOLLOWUP_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
