"""kframes_gen.py — 保存中间采样步的解码帧图像(2026-10-18)

对 cow 9200↔9250 的四条臂(base/donor/ll@1.0/hh@1.0), 在检查点 k∈{1,3,8,17,50}
解码 x̂₀ 到像素并保存 PNG。这些帧展示采样过程中牛从噪声中"浮现"的实际视觉效果。
"""
import json, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from model_config import WAN22_MODEL_PATH
sys.path.insert(0, str(HERE))

from ll_blend import blend_eps
from ot_common import latent_shape_ot

SHAPE = latent_shape_ot(81, 480, 832)
MODEL = WAN22_MODEL_PATH
CKPTS = (1, 3, 8, 17, 50)
SB, SD = 9200, 9250


def sample_noise(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(SHAPE, generator=g)


def run_and_decode(pipe, eps, tag, outdir):
    """跑 50 步, 在检查点解码 x̂₀ 并保存帧 [0, 40, 80]。"""
    rec = {"i": 0, "x0s": {}, "final": None}
    real_step = pipe.scheduler.step

    def spy(model_output, timestep, sample, **kw):
        rec["i"] += 1
        if rec["i"] in CKPTS:
            sig = float(pipe.scheduler.sigmas[rec["i"] - 1])
            rec["x0s"][rec["i"]] = (sample.detach()[0].float().cpu(),
                                    model_output.detach()[0].float().cpu(), sig)
        out = real_step(model_output, timestep, sample, **kw)
        if rec["i"] == 50:
            rec["final"] = out[0].detach()[0].float().cpu()  # 真正的最终潜变量
        return out

    pipe.scheduler.step = spy
    try:
        from PIL import Image
        pipe(image=Image.open(HERE / "payload/donor_cow_p_cond.png"),
             prompt="a cow standing in a green pasture",
             height=480, width=832, num_frames=81,
             num_inference_steps=50, guidance_scale=5.0,
             latents=eps.unsqueeze(0), output_type="latent")
    finally:
        pipe.scheduler.step = real_step

    cfg = pipe.vae.config
    for k in CKPTS:
        if k == 50:
            continue  # k=50 从已有视频文件取帧(见 main), 不手动解码
        else:
            s, v, sig = rec["x0s"][k]
            z = (s - sig * v).numpy()
        # un-scale
        std = torch.tensor(cfg.latents_std, dtype=torch.float32).reshape(1, -1, 1, 1, 1)
        mean = torch.tensor(cfg.latents_mean, dtype=torch.float32).reshape(1, -1, 1, 1, 1)
        z_t = torch.from_numpy(z).unsqueeze(0)
        z_t = z_t / std + mean
        # decode
        with torch.no_grad():
            vid = pipe.vae.decode(z_t.to(pipe.vae.dtype).to(pipe.device),
                                   return_dict=False)[0]
            frames = (pipe.video_processor.postprocess_video(vid, output_type="np")[0] * 255).round().astype(np.uint8)
        # 保存 f0, f40, f80
        for f_idx in (0, 40, 80):
            out = outdir / f"{tag}_k{k:02d}_f{f_idx:02d}.png"
            Image.fromarray(frames[f_idx]).save(out)
        print(f"  [{tag}] k={k}: saved f0/f40/f80", flush=True)


def main():
    from diffusers import WanImageToVideoPipeline
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.vae.enable_tiling(); pipe.to("cuda")

    outdir = HERE / "paper" / "kframes"
    outdir.mkdir(parents=True, exist_ok=True)

    eb, ed = sample_noise(SB), sample_noise(SD)
    arms = [
        ("base", eb),
        ("donor", ed),
        ("ll100", blend_eps(eb, ed, 1.0, band="ll")),
        ("hh100", blend_eps(eb, ed, 1.0, band="hh")),
    ]
    for tag, eps in arms:
        existing = list(outdir.glob(f"{tag}_k*"))
        if len(existing) >= len(CKPTS) * 3:
            print(f"[skip] {tag}", flush=True); continue
        print(f"=== {tag} ===", flush=True)
        t0 = time.time()
        run_and_decode(pipe, eps, tag, outdir)
        print(f"  {time.time()-t0:.0f}s", flush=True)
        # k=50: 从最终视频文件直接取帧(零误差)
    import imageio.v3 as iio
    from PIL import Image
    vid_map = {'base': 'videos/cow_p_s9200_ctrlB.mp4', 'donor': 'videos/cow_p_s9250_ctrlB.mp4', 'll100': 'lambda_scan/i2v_lam_ll100_s9200.mp4', 'hh100': 'lambda_scan/i2v_lam_hh100_s9200.mp4'}
    for tag, vid_path in vid_map.items():
        vp = HERE / vid_path
        if not vp.exists():
            print(f"[warn] video not found: {vp}"); continue
        for f_idx in (0, 40, 80):
            out = outdir / f"{tag}_k50_f{f_idx:02d}.png"
            if out.exists(): continue
            arr = np.asarray(iio.imread(str(vp), index=f_idx))[..., :3]
            Image.fromarray(arr).save(out)
        print(f"  [{tag}] k=50 frames from video", flush=True)
    print("KFRAMES_DONE", flush=True)


if __name__ == "__main__":
    main()
