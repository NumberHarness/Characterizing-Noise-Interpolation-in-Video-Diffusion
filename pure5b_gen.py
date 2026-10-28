"""pure5b_gen.py — 同模型纯 T2V lambda 扫描(Wan2.2-TI2V-5B + WanPipeline, 2026-10-17)

关键: 用 WanPipeline(非 WanImageToVideoPipeline)加载同一 5B 权重, 不传 image。
帧 0 不再被条件图占据 → 全部 21 帧 = 噪声, 全部参与混合(无帧 0 保留)。
同模型、同种子、同公式 → 无跨模型混淆变量。
预注册 pure5b_manifest.json; 幂等; 逐臂多牛守卫。
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
from ot_common import whiteness_gate

SHAPE = (48, 21, 30, 52)   # 同 TI2V-5B(同 VAE)
MODEL = WAN22_MODEL_PATH
LAMS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
MODES = ("field", "ll", "lp")
PAIRS = [("cow_p", "a cow standing in a green pasture", 9220, 9240),
         ("horse_f", "a horse galloping across an open green field", 9300, 9240)]


def blend_all(eps_b, eps_d, lam, band, level=1, wave="haar", frac=0.25):
    """T2V 混合: 全部帧(含帧 0)参与混合, 无条件帧保留。"""
    from ll_blend import lp_mask
    assert 0.0 <= lam <= 1.0
    if lam == 0.0:
        return eps_b.clone()
    if lam == 1.0 and band == "field":
        return eps_d.clone()
    b = eps_b.clone().float()
    d = eps_d.clone().float()
    _, T, H, W = b.shape
    norm = math.sqrt((1 - lam) ** 2 + lam ** 2)
    if band == "field":
        out = (b + lam * (d - b)) / norm
    elif band == "ll":
        import pywt
        out = b.clone()
        e, dn = b.numpy(), d.numpy()
        C = b.shape[0]
        for c in range(C):
            for t in range(T):                    # ← 全部帧, 含 t=0
                cb = pywt.wavedec2(e[c, t], wave, mode="periodization", level=level)
                cd = pywt.wavedec2(dn[c, t], wave, mode="periodization", level=level)
                cb[0][:] = (cb[0] + lam * (cd[0] - cb[0])) / norm
                out[c, t] = torch.from_numpy(pywt.waverec2(
                    cb, wave, mode="periodization")[:H, :W].astype(np.float32))
    elif band == "lp":
        m, _ = lp_mask(H, W, frac)
        fb = np.fft.rfft2(b.numpy(), axes=(-2, -1), norm="ortho")
        fd = np.fft.rfft2(d.numpy(), axes=(-2, -1), norm="ortho")
        fb[..., m] = (fb[..., m] + lam * (fd[..., m] - fb[..., m])) / norm
        out = torch.from_numpy(np.fft.irfft2(
            fb, s=(H, W), axes=(-2, -1), norm="ortho").astype(np.float32))
    else:
        raise KeyError(band)
    return out                               # ← 无 out[:, 0] 保留


def jobs():
    out = []
    for scene, prompt, sb, sd in PAIRS:
        d = HERE / "pure5b" / scene
        out.append({"kind": "ctrl", "scene": scene, "prompt": prompt, "seed": sb,
                    "mp4": d / f"ctrl_s{sb}.mp4"})
        out.append({"kind": "ctrl", "scene": scene, "prompt": prompt, "seed": sd,
                    "mp4": d / f"ctrl_s{sd}.mp4"})
        for band in MODES:
            for lam in LAMS:
                out.append({"kind": "arm", "scene": scene, "prompt": prompt,
                            "band": band, "lam": lam, "sb": sb, "sd": sd,
                            "mp4": d / f"{band}{int(round(lam*100)):02d}_s{sb}d{sd}.mp4"})
    return out


def ensure_manifest():
    p = HERE / "pure5b_manifest.json"
    if not p.exists():
        p.write_text(json.dumps({
            "design": "同模型纯 T2V(WanPipeline, 无图)lambda 扫描, cow+horse, 全帧混合",
            "model": "Wan2.2-TI2V-5B(同权重, WanPipeline 替代 WanImageToVideoPipeline)",
            "key_diff": "帧 0 参与混合(无条件帧); 21 帧全部可编辑",
            "arms": [{"kind": j["kind"], "band": j.get("band"), "lam": j.get("lam")} for j in jobs()],
            "gates": "白性门; 多牛守卫"}, ensure_ascii=False, indent=1))
        print(f"[prereg] {p} 落盘", flush=True)
    return p


def sample_noise(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(SHAPE, generator=g)


def main():
    ensure_manifest()
    from diffusers import WanPipeline
    from diffusers.utils import export_to_video
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                        low_cpu_mem_usage=True)
    pipe.vae.enable_tiling(); pipe.to("cuda")
    for j in jobs():
        if j["mp4"].exists() and j["mp4"].stat().st_size > 0:
            print(f"[skip] {j['mp4'].name}", flush=True); continue
        j["mp4"].parent.mkdir(parents=True, exist_ok=True)
        if j["kind"] == "ctrl":
            eps = sample_noise(j["seed"])
        else:
            eb, ed = sample_noise(j["sb"]), sample_noise(j["sd"])
            eps = blend_all(eb, ed, j["lam"], band=j["band"])
        g = whiteness_gate(eps)
        t0 = time.time()
        o = pipe(prompt=j["prompt"], height=480, width=832, num_frames=81,
                 num_inference_steps=50, guidance_scale=5.0,
                 latents=eps.unsqueeze(0), output_type="np")
        export_to_video(o.frames[0], str(j["mp4"]), fps=16)
        del o; torch.cuda.empty_cache()
        print(f"[{j['scene']} {j.get('band') or 'ctrl'}@{j.get('lam') or j.get('seed','')}] "
              f"白性={g['pass']}(std {g['std']}) {time.time()-t0:.0f}s", flush=True)
    print("PURE5B_DONE", flush=True)


if __name__ == "__main__":
    main()
