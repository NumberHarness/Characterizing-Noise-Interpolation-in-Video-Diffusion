"""kresp_run.py — k 步有效响应批(2026-10-15, 预注册 kresp_manifest.json)

设计: 每对跑 6 条完整 50 步轨迹(base/donor ctrl + field@.5 + ll/lp/hh@1.0),
scheduler.step 间谍在检查点 k∈{1,2,3,5,8,12,17,25,35,50} 记录步后潜变量;
ρ_arm(k) = ⟨x_k(arm)−x_k(base), e₅₀⟩/‖e₅₀‖², e₅₀ = donor 终态 − base 终态。
donor 自身即 field@1.0 曲线(ρ_donor(50)≡1)。全程无 VAE 解码(output_type=latent),
无线性假设。验证: ρ_*(50) 应对表行为层 R_LL/R_lp; ρ_hh 为刹车直接读出。
用法: python3 kresp_run.py pilot | all
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
from model_config import WAN22_MODEL_PATH
sys.path.insert(0, str(HERE))

from ll_blend import blend_eps
from ot_common import latent_shape_ot, whiteness_gate

SHAPE = latent_shape_ot(81, 480, 832)
CKPTS = (1, 2, 3, 5, 8, 12, 17, 25, 35, 50)
MODEL = WAN22_MODEL_PATH
SCENES = {
    "cow_p": ("a cow standing in a green pasture", HERE / "payload/donor_cow_p_cond.png"),
    "horse_f": ("a horse galloping across an open green field", HERE / "scenes/horse_f/cond.png"),
    "boat_l": ("a sailboat sailing across a calm blue lake", HERE / "scenes/boat_l/cond.png"),
}
PAIRS = {"pilot": [("cow_p", 9200, 9250)],
         "all": [("cow_p", 9200, 9250), ("cow_p", 9220, 9240),
                 ("horse_f", 9300, 9240), ("boat_l", 9250, 9210)]}
ARMS = [("field", 0.5), ("ll", 1.0), ("lp", 1.0), ("hh", 1.0)]


def sample_noise(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(SHAPE, generator=g)


def run_traj(pipe, image, prompt, eps, tag):
    """一条 50 步轨迹; 间谍记录检查点步后潜变量(cpu fp32)。"""
    ck = {}
    state = {"i": 0}
    real_step = pipe.scheduler.step

    def spy(model_output, timestep, sample, **kw):
        out = real_step(model_output, timestep, sample, **kw)
        state["i"] += 1
        if state["i"] in CKPTS:
            ck[state["i"]] = out[0].detach()[0].float().cpu()
        return out

    pipe.scheduler.step = spy
    try:
        pipe(image=image, prompt=prompt, height=480, width=832, num_frames=81,
             num_inference_steps=50, guidance_scale=5.0,
             latents=eps.unsqueeze(0), output_type="latent")
    finally:
        pipe.scheduler.step = real_step
    assert sorted(ck) == list(CKPTS), f"{tag}: 检查点 {sorted(ck)}"
    return ck


def curves(base_ck, donor_ck, arms_ck):
    e50 = donor_ck[50] - base_ck[50]
    n50 = float((e50 * e50).sum())
    out = {}
    for name, ck in [("field@1(donor)", donor_ck)] + list(arms_ck.items()):
        out[name] = {str(k): round(float(((ck[k] - base_ck[k]) * e50).sum() / n50), 4)
                     for k in CKPTS}
    return out


def main(mode, force=False):
    from PIL import Image
    from diffusers import WanImageToVideoPipeline
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.vae.enable_tiling()
    pipe.to("cuda")
    for scene, sb, sd in PAIRS[mode]:
        outp = HERE / f"kresp_{sb}d{sd}.json"
        if outp.exists() and not force:
            print(f"[skip] {outp.name}", flush=True)
            continue
        prompt, cond = SCENES[scene]
        img = Image.open(cond)
        eb, ed = sample_noise(sb), sample_noise(sd)
        base_ck = run_traj(pipe, img, prompt, eb, f"{sb} base")
        donor_ck = run_traj(pipe, img, prompt, ed, f"{sd} donor")
        arms_ck = {}
        for band, lam in ARMS:
            eps = blend_eps(eb, ed, lam, band=band)
            g = whiteness_gate(eps[:, 1:])
            arms_ck[f"{band}@{lam}"] = run_traj(pipe, img, prompt, eps, f"{band}{lam}")
            print(f"[{sb}↔{sd} {band}@{lam}] 白性={g['pass']}(std {g['std']})", flush=True)
        cur = curves(base_ck, donor_ck, arms_ck)
        outp.write_text(json.dumps(cur, ensure_ascii=False, indent=1))
        for name, c in cur.items():
            pts = " ".join(f"k{k}:{c[str(k)]:+.3f}" for k in (3, 8, 17, 35, 50))
            print(f"[{sb}↔{sd}] {name:16s} {pts}", flush=True)
    print("KRESP_DONE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "all"), nargs="?", default="pilot")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing sampling-step measurement JSON")
    args = parser.parse_args()
    main(args.mode, force=args.force)
