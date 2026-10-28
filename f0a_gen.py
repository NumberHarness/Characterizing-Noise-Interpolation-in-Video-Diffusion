"""Generate the paper's fixed f0A controls and interpolation arms."""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

import arm_catalog as catalog
from ll_blend import blend_eps
from model_config import WAN22_MODEL_PATH
from ot_common import latent_shape_ot, whiteness_gate

ROOT = Path(__file__).resolve().parent
SHAPE = latent_shape_ot(81, 480, 832)

# These auxiliary controls provide the frame-40 conditioning images used for f0B.
AUX_F0B_SEEDS = {"cow_p": 9230, "horse_f": 9220, "boat_l": 9240}
F0A_FIELD100_PAIRS = ("cow2", "boat")


def sample_noise(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(SHAPE, generator=generator)


def control_path(scene: str, seed: int) -> Path:
    if scene == "cow_p":
        return ROOT / "videos" / f"cow_p_s{seed}_ctrlB.mp4"
    return ROOT / "scenes" / scene / f"{scene}_s{seed}_ctrlB.mp4"


def seeds_by_scene() -> dict[str, set[int]]:
    seeds = {scene: {AUX_F0B_SEEDS[scene]} for scene in catalog.SCENES}
    for pair in catalog.PAIRS.values():
        seeds[pair["scene"]].update((pair["sb"], pair["sd"]))
    return seeds


def require_inputs(scene: str) -> tuple[str, Path]:
    prompt = catalog.PROMPTS[scene]
    pair = next(name for name, spec in catalog.PAIRS.items()
                if spec["scene"] == scene)
    image = catalog.condition_image("f0A", pair)
    if image is None or not image.is_file():
        raise FileNotFoundError(
            f"Missing f0A conditioning image for {scene}: {image}")
    return prompt, image


def generate_scene(scene: str) -> None:
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    from PIL import Image

    prompt, image_path = require_inputs(scene)
    image = Image.open(image_path).convert("RGB")
    pipe = None
    try:
        for seed in sorted(seeds_by_scene()[scene]):
            path = control_path(scene, seed)
            if path.is_file() and path.stat().st_size:
                print(f"[skip] f0A control {scene} seed={seed}", flush=True)
                continue
            if pipe is None:
                pipe = WanImageToVideoPipeline.from_pretrained(
                    WAN22_MODEL_PATH,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                pipe.vae.enable_tiling()
                pipe.to("cuda")
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
                latents=sample_noise(seed).unsqueeze(0),
                output_type="np",
            )
            export_to_video(output.frames[0], str(path), fps=16)
            del output
            torch.cuda.empty_cache()
            print(f"[control {scene} seed={seed}] {time.time() - start:.0f}s",
                  flush=True)

        for pair_name, pair in catalog.PAIRS.items():
            if pair["scene"] != scene:
                continue
            eps_base = sample_noise(pair["sb"])
            eps_donor = sample_noise(pair["sd"])
            for mode in catalog.MODES:
                for lam in catalog.LAMS:
                    if mode == "field" and lam == 1.0:
                        # Full-field lambda=1 is exactly the donor control.
                        continue
                    path = catalog.arm_path("f0A", pair_name, mode, lam)
                    if path.is_file() and path.stat().st_size:
                        print(f"[skip] {pair_name} {mode}@{lam:g}", flush=True)
                        continue
                    if pipe is None:
                        pipe = WanImageToVideoPipeline.from_pretrained(
                            WAN22_MODEL_PATH,
                            torch_dtype=torch.bfloat16,
                            low_cpu_mem_usage=True,
                        )
                        pipe.vae.enable_tiling()
                        pipe.to("cuda")
                    eps = blend_eps(
                        eps_base, eps_donor, lam, band=mode
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
                    print(
                        f"[{pair_name} {mode}@{lam:g}] "
                        f"whiteness={gate['pass']} "
                        f"({time.time() - start:.0f}s)",
                        flush=True,
                    )

        # Preserve the separately measured field@1 anchors in the paper readout.
        for pair_name in F0A_FIELD100_PAIRS:
            pair = catalog.PAIRS[pair_name]
            if pair["scene"] != scene:
                continue
            path = catalog.field100_path("f0A", pair_name)
            if path is None:
                continue
            if path.is_file() and path.stat().st_size:
                print(f"[skip] {pair_name} field@1 anchor", flush=True)
                continue
            if pipe is None:
                pipe = WanImageToVideoPipeline.from_pretrained(
                    WAN22_MODEL_PATH,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                pipe.vae.enable_tiling()
                pipe.to("cuda")
            eps = sample_noise(pair["sd"]).unsqueeze(0)
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
            print(
                f"[{pair_name} field@1 anchor] whiteness={gate['pass']} "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )
    finally:
        if pipe is not None:
            del pipe
            gc.collect()
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=tuple(catalog.SCENES))
    args = parser.parse_args()
    scenes = (args.scene,) if args.scene else tuple(catalog.SCENES)
    for scene in scenes:
        print(f"=== f0A generation: {scene} ===", flush=True)
        generate_scene(scene)


if __name__ == "__main__":
    main()
