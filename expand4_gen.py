"""expand4_gen.py — 四组扩展批: 两个新对 × 四条件(2026-10-18)

新对: cow 9200↔9250(原始基线, sep 312) + boat 9250↔9210(帆船, sep 420)
四条件: f0A(I2V 信息图) / f0B(I2V 另一图) / gray(灰图) / pure(纯 T2V)
105 条缺口, 幂等, 分 I2V/T2V 两段(管线不同)。
预注册 expand4_manifest.json。
"""
import json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from model_config import WAN22_MODEL_PATH
sys.path.insert(0, str(HERE))
from ll_blend import lp_mask
from ot_common import whiteness_gate

SHAPE = (48, 21, 30, 52)
MODEL = WAN22_MODEL_PATH
LAMS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
MODES = ("field", "ll", "lp")

SCENES = {
    "cow_p": "a cow standing in a green pasture",
    "boat_l": "a sailboat sailing across a calm blue lake",
}
NEW_PAIRS = [("cow_p", 9200, 9250), ("boat_l", 9250, 9210)]


def blend(eps_b, eps_d, lam, band, t2v=False):
    """t2v=True 时帧 0 也混合; 否则帧 0 保留 base。"""
    from ll_blend import blend_eps
    if not t2v:
        return blend_eps(eps_b, eps_d, lam, band=band)
    # T2V: 全帧混合
    if lam == 0.0: return eps_b.clone()
    if lam == 1.0 and band == "field": return eps_d.clone()
    b, d = eps_b.clone().float(), eps_d.clone().float()
    _, T, H, W = b.shape
    norm = math.sqrt((1-lam)**2 + lam**2)
    if band == "field":
        return (b + lam*(d-b)) / norm
    if band == "ll":
        import pywt
        out = b.clone()
        for c in range(b.shape[0]):
            for t in range(T):
                cb = pywt.wavedec2(b.numpy()[c,t], "haar", mode="periodization", level=1)
                cd = pywt.wavedec2(d.numpy()[c,t], "haar", mode="periodization", level=1)
                cb[0][:] = (cb[0] + lam*(cd[0]-cb[0])) / norm
                out[c,t] = torch.from_numpy(pywt.waverec2(cb, "haar", mode="periodization")[:H,:W].astype(np.float32))
        return out
    if band == "lp":
        m, _ = lp_mask(H, W, 0.25)
        fb = np.fft.rfft2(b.numpy(), axes=(-2,-1), norm="ortho")
        fd = np.fft.rfft2(d.numpy(), axes=(-2,-1), norm="ortho")
        fb[..., m] = (fb[..., m] + lam*(fd[..., m]-fb[..., m])) / norm
        return torch.from_numpy(np.fft.irfft2(fb, s=(H,W), axes=(-2,-1), norm="ortho").astype(np.float32))


def sample_noise(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(SHAPE, generator=g)


def build_jobs():
    """返回 [(pipeline_type, scene, sb, sd, kind, band, lam, mp4_path, cond_path)]"""
    jobs = []
    for scene, sb, sd in NEW_PAIRS:
        prompt = SCENES[scene]

        # ── Group ①: f0A 补缺 ──
        d = HERE/"lambda_scan" if scene=="cow_p" else HERE/"scenes"/scene
        cond = HERE/"payload/donor_cow_p_cond.png" if scene=="cow_p" else HERE/"scenes"/scene/"cond.png"
        for band in MODES:
            for lam in LAMS:
                if band=="field" and lam==1.0: continue
                tag = f"i2v_lam_{band}{int(round(lam*100)):02d}_s{sb}" + \
                      ("" if scene=="cow_p" else f"d{sd}")
                mp4 = d/f"{tag}.mp4"
                if not mp4.exists():
                    jobs.append(("i2v", scene, sb, sd, "arm", band, lam, mp4, cond))

        # ── Group ②: f0B 全新 ──
        d2 = HERE/"ga2"/scene
        cond2 = HERE/"groups"/"f0B"/f"{scene}.png"
        mp4 = d2/f"ctrlB_f0B_s{sb}.mp4"
        if not mp4.exists():
            jobs.append(("i2v", scene, sb, sd, f"ctrl_s{sb}", None, None, mp4, cond2))
        for band in MODES:
            for lam in LAMS:
                if band=="field" and lam==1.0: continue
                mp4 = d2/f"{band}{int(round(lam*100)):02d}_s{sb}d{sd}.mp4"
                if not mp4.exists():
                    jobs.append(("i2v", scene, sb, sd, "arm", band, lam, mp4, cond2))

        # ── Group ③: gray 补缺 ──
        d3 = HERE/"x0" if scene=="cow_p" else HERE/"t2v"/scene
        cond3 = HERE/"x0"/"cond_gray.png"
        for s in (sb, sd):
            mp4 = d3/f"ctrl_s{s}.mp4"
            if not mp4.exists():
                jobs.append(("i2v", scene, sb, sd, f"ctrl_s{s}", None, None, mp4, cond3))
        for band in MODES:
            for lam in LAMS:
                if band=="field" and lam==1.0: continue
                mp4 = d3/f"{band}{int(round(lam*100)):02d}_s{sb}d{sd}.mp4"
                if not mp4.exists():
                    jobs.append(("i2v", scene, sb, sd, "arm", band, lam, mp4, cond3))

        # ── Group ④: pure T2V 全新 ──
        d4 = HERE/"pure5b"/scene
        for s in (sb, sd):
            mp4 = d4/f"ctrl_s{s}.mp4"
            if not mp4.exists():
                jobs.append(("t2v", scene, sb, sd, f"ctrl_s{s}", None, None, mp4, None))
        for band in MODES:
            for lam in LAMS:
                if band=="field" and lam==1.0: continue
                mp4 = d4/f"{band}{int(round(lam*100)):02d}_s{sb}d{sd}.mp4"
                if not mp4.exists():
                    jobs.append(("t2v", scene, sb, sd, "arm", band, lam, mp4, None))

    return jobs


def main(stage=None):
    all_jobs = build_jobs()
    if stage is None:
        (HERE/"expand4_manifest.json").write_text(json.dumps({
            "design": "四组扩展: cow 9200↔9250 + boat 9250↔9210 × {f0A,f0B,gray,pure}",
            "total_gaps": len(all_jobs)}, ensure_ascii=False, indent=1))
        print(f"[prereg] expand4_manifest.json ({len(all_jobs)} gaps)", flush=True)
        import subprocess
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        for s in ("i2v", "t2v"):
            r = subprocess.run([sys.executable, __file__, s], env=env)
            if r.returncode != 0: sys.exit(r.returncode)
        print("EXPAND4_DONE", flush=True)
        return

    # 按管线类型执行
    jobs = [j for j in all_jobs if j[0] == stage]
    print(f"=== {stage}: {len(jobs)} 条 ===", flush=True)
    from PIL import Image
    from diffusers.utils import export_to_video

    if stage == "i2v":
        from diffusers import WanImageToVideoPipeline
        pipe_by_scene = {}
    else:
        from diffusers import WanPipeline
        pipe_by_scene = {}

    for ptype, scene, sb, sd, kind, band, lam, mp4, cond in jobs:
        if mp4.exists() and mp4.stat().st_size > 0:
            print(f"[skip] {mp4}", flush=True); continue
        mp4.parent.mkdir(parents=True, exist_ok=True)

        if scene not in pipe_by_scene:
            # 先释放旧场景的管线(防 OOM)
            if pipe_by_scene:
                del pipe
            pipe_by_scene.clear()
            import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

            if stage == "i2v":
                pipe = WanImageToVideoPipeline.from_pretrained(
                    MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
            else:
                pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                    low_cpu_mem_usage=True)
            pipe.vae.enable_tiling(); pipe.to("cuda")
            pipe_by_scene[scene] = pipe
        pipe = pipe_by_scene[scene]

        if kind.startswith("ctrl"):
            seed = int(kind.split("_s")[1])
            eps = sample_noise(seed)
        else:
            eb, ed = sample_noise(sb), sample_noise(sd)
            eps = blend(eb, ed, lam, band=band, t2v=(stage=="t2v"))

        g = whiteness_gate(eps)
        t0 = time.time()
        if stage == "i2v":
            o = pipe(image=Image.open(cond), prompt=SCENES[scene],
                     height=480, width=832, num_frames=81,
                     num_inference_steps=50, guidance_scale=5.0,
                     latents=eps.unsqueeze(0), output_type="np")
        else:
            o = pipe(prompt=SCENES[scene], height=480, width=832, num_frames=81,
                     num_inference_steps=50, guidance_scale=5.0,
                     latents=eps.unsqueeze(0), output_type="np")
        export_to_video(o.frames[0], str(mp4), fps=16)
        del o; torch.cuda.empty_cache()
        print(f"[{scene} {kind}{'@'+str(band)+str(lam) if band else ''}] "
              f"白性={g['pass']} {time.time()-t0:.0f}s", flush=True)

    print(f"_{stage.upper()}_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
