"""ga2_gen.py — f0_B 世界重扫批(2026-10-17, 预注册 ga2_manifest.json)

动机: GA 参照版曲线与 f0_A 版逐位相同(臂=同批视频, 设计退化, 用户指出)。本批把
两对刹车案例(cow 9220↔9240, horse 9300↔9240)的 λ 扫描整体搬到 f0_B 世界:
base ctrl@f0B + field/ll/lp × λ{.25,.5,.75,.9} + donor ctrl@f0B(已有) = 每对 13 条。
问题: 剂量-响应结构(翻转窗口/超调/刹车)是否随生成条件迁移; 逐臂多牛守卫。
"""
import json, sys, time
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
from model_config import WAN22_MODEL_PATH
sys.path.insert(0, str(HERE))
from ll_blend import blend_eps
from ot_common import latent_shape_ot, whiteness_gate
SHAPE = latent_shape_ot(81, 480, 832)
PAIRS = [("cow_p", "a cow standing in a green pasture", 9220, 9240),
         ("horse_f", "a horse galloping across an open green field", 9300, 9240)]
LAMS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
MODES = ("field", "ll", "lp")


def jobs():
    out = []
    for scene, prompt, sb, sd in PAIRS:
        d = HERE / "ga2" / scene
        out.append({"kind": "ctrl", "scene": scene, "prompt": prompt, "sb": sb, "sd": sd,
                    "mp4": d / f"ctrlB_f0B_s{sb}.mp4"})
        for band in MODES:
            for lam in LAMS:
                out.append({"kind": "arm", "scene": scene, "prompt": prompt, "band": band,
                            "lam": lam, "sb": sb, "sd": sd,
                            "mp4": d / f"{band}{int(round(lam*100)):02d}_s{sb}d{sd}.mp4"})
    return out


def ensure_manifest():
    p = HERE / "ga2_manifest.json"
    if not p.exists():
        p.write_text(json.dumps({
            "design": "f0_B 世界重扫: 两刹车案例整条 λ 扫描搬到 donor 参照世界(f0_B)",
            "arms": [{"kind": j["kind"], "band": j.get("band"), "lam": j.get("lam"),
                      "pair": f"{j['sb']}↔{j['sd']}"} for j in jobs()],
            "predictions": {
                "P-c1 条件迁移": "翻转窗口与超调结构在 f0_B 下复现(种子地址+剂量-响应"
                                "对生成条件稳健), 或给出量化偏移",
                "P-c2 刹车迁移": "horse 的 field 中段钉死 + ll/lp 翻转形态在 f0_B 存活"
                                "(刹车是噪声对属性而非条件属性)",
                "P-c3 多牛": "f0_B 信息性在场 ⇒ 中剂量无实体分裂(对照灰图@0.5)"},
            "gates": "白性门逐臂; 逐臂多牛守卫(ent 协议); 幂等"}, ensure_ascii=False, indent=1))
        print(f"[prereg] {p} 落盘", flush=True)
    return p


def sample_noise(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((1, *SHAPE), generator=g)


def stage(scene):
    from PIL import Image
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    js = [j for j in jobs() if j["scene"] == scene]
    cond = HERE / "groups" / "f0B" / f"{scene}.png"
    pipe = None
    for j in js:
        if j["mp4"].exists() and j["mp4"].stat().st_size > 0:
            print(f"[skip] {j['mp4'].name}", flush=True); continue
        if pipe is None:
            pipe = WanImageToVideoPipeline.from_pretrained(
                WAN22_MODEL_PATH,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
            pipe.vae.enable_tiling(); pipe.to("cuda")
        j["mp4"].parent.mkdir(parents=True, exist_ok=True)
        if j["kind"] == "ctrl":
            eps = sample_noise(j["sb"])
        else:
            eps = blend_eps(sample_noise(j["sb"])[0], sample_noise(j["sd"])[0],
                            j["lam"], band=j["band"]).unsqueeze(0)
        g = whiteness_gate(eps[0, :, 1:])
        t0 = time.time()
        o = pipe(image=Image.open(cond), prompt=j["prompt"], height=480, width=832,
                 num_frames=81, num_inference_steps=50, guidance_scale=5.0,
                 latents=eps, output_type="np")
        export_to_video(o.frames[0], str(j["mp4"]), fps=16)
        del o; torch.cuda.empty_cache()
        print(f"[{scene} {j.get('band') or 'ctrl'}@{j.get('lam') or ''}] "
              f"白性={g['pass']}(std {g['std']}) {time.time()-t0:.0f}s", flush=True)


def main():
    ensure_manifest()
    import os, subprocess
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for scene in ("horse_f", "cow_p"):
        r = subprocess.run([sys.executable, __file__, scene], env=env)
        if r.returncode != 0:
            sys.exit(r.returncode)
    print("GA2_GEN_DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("cow_p", "horse_f"):
        stage(sys.argv[1])
    else:
        main()
