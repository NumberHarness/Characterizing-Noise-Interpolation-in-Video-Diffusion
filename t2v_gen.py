"""t2v_gen.py — 纯 T2V(灰图=零信息条件)lambda 扫描: horse 补齐(2026-10-17)

设计: base ctrl / donor ctrl / 全部 λ 臂共享同一张纯灰图(视觉信息=零),
模型仅凭 prompt + 噪声生成完整视频。cow 数据已有(x0 批), 本批补 horse 9300↔9240。
预注册 t2v_manifest.json; 幂等; 逐臂多牛守卫。
"""
import json, os, sys, time
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
from model_config import WAN22_MODEL_PATH
sys.path.insert(0, str(HERE))
from ll_blend import blend_eps
from ot_common import latent_shape_ot, whiteness_gate
SHAPE = latent_shape_ot(81, 480, 832)
PAIR = ("horse_f", "a horse galloping across an open green field", 9300, 9240)
LAMS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
MODES = ("field", "ll", "lp")


def jobs():
    scene, prompt, sb, sd = PAIR
    d = HERE / "t2v" / scene
    out = [{"kind": "ctrl", "seed": sb, "mp4": d / f"ctrl_s{sb}.mp4"},
           {"kind": "ctrl", "seed": sd, "mp4": d / f"ctrl_s{sd}.mp4"}]
    for band in MODES:
        for lam in LAMS:
            out.append({"kind": "arm", "band": band, "lam": lam,
                        "mp4": d / f"{band}{int(round(lam*100)):02d}_s{sb}d{sd}.mp4"})
    return out


def ensure_manifest():
    p = HERE / "t2v_manifest.json"
    if not p.exists():
        p.write_text(json.dumps({
            "design": "纯 T2V(灰图=零信息)horse 9300↔9240 lambda 扫描",
            "arms": [{"kind": j["kind"], "band": j.get("band"), "lam": j.get("lam")} for j in jobs()],
            "predictions": {
                "P-t1 可行性": "灰图下 horse 模型可生成可检测的马(ctrl ≥6/9 有效帧)",
                "P-t2 剂量响应": "三段式(死区→阈值→平台)在无信息条件下仍成立",
                "P-t3 实体分裂": "中剂量(λ≈0.5)可能出现多实体(与 cow 灰图一致)"},
            "gates": "白性门; 帧0=灰图(不触); 多牛守卫"}, ensure_ascii=False, indent=1))
        print(f"[prereg] {p} 落盘", flush=True)
    return p


def sample_noise(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((1, *SHAPE), generator=g)


def main():
    ensure_manifest()
    from PIL import Image
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    gray = HERE / "x0" / "cond_gray.png"
    img = Image.open(gray)
    pipe = WanImageToVideoPipeline.from_pretrained(
        WAN22_MODEL_PATH,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.vae.enable_tiling(); pipe.to("cuda")
    for j in jobs():
        if j["mp4"].exists() and j["mp4"].stat().st_size > 0:
            print(f"[skip] {j['mp4'].name}", flush=True); continue
        j["mp4"].parent.mkdir(parents=True, exist_ok=True)
        if j["kind"] == "ctrl":
            eps = sample_noise(j["seed"])
        else:
            eps = blend_eps(sample_noise(PAIR[2])[0], sample_noise(PAIR[3])[0],
                            j["lam"], band=j["band"]).unsqueeze(0)
        g = whiteness_gate(eps[0, :, 1:])
        t0 = time.time()
        o = pipe(image=img, prompt=PAIR[1], height=480, width=832, num_frames=81,
                 num_inference_steps=50, guidance_scale=5.0, latents=eps, output_type="np")
        export_to_video(o.frames[0], str(j["mp4"]), fps=16)
        del o; torch.cuda.empty_cache()
        print(f"[{j.get('band') or 'ctrl'}@{j.get('lam') or ''}] "
              f"白性={g['pass']}(std {g['std']}) {time.time()-t0:.0f}s", flush=True)
    print("T2V_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
