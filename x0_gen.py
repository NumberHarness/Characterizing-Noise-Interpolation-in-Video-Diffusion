"""x0_gen.py — T2V 域(无信息条件图)三模式对比批(2026-10-14, 预注册 x0_manifest.json)

设计: 同模型(Wan2.2-TI2V-5B) + 纯灰条件图(832×480, RGB 128)——只动"条件信息量"一个变量。
对: cow 两对(9200↔9250 现象对, 9220↔9240 阴性复制); 模式 field/ll/lp × λ{0.1,0.25,0.5}
(+ll/lp@1.0; field@1.0 ≡ donor ctrl, 确定性, 不重复生成)。共 26 条。
风险门(预注册): base/donor ctrl 主实体 ≥6/9 帧有效, 否则"灰图域不可行"如实报告。
指标(M1/M2/M3)定义见 manifest; M3 皮相 = 跨视频 DINOv2 亲和度(嵌合体检验)。
"""
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
from model_config import WAN22_MODEL_PATH
sys.path.insert(0, str(HERE))

from ll_blend import blend_eps
from ot_common import latent_shape_ot, whiteness_gate

SHAPE = latent_shape_ot(81, 480, 832)
PAIRS = [(9200, 9250), (9220, 9240)]
ARMS = [("field", 0.1), ("field", 0.25), ("field", 0.5),
        ("ll", 0.1), ("ll", 0.25), ("ll", 0.5), ("ll", 1.0),
        ("lp", 0.1), ("lp", 0.25), ("lp", 0.5), ("lp", 1.0)]
OUT = HERE / "x0"
PROMPT = "a cow standing in a green pasture"


def ensure_cond():
    OUT.mkdir(exist_ok=True)
    p = OUT / "cond_gray.png"
    if not p.exists():
        from PIL import Image
        Image.new("RGB", (832, 480), (128, 128, 128)).save(p)
    return p


def jobs():
    """ctrl 2 条/对 + 11 臂/对 = 26; 命名避开 i2v_lam_ 前缀(防 phase1 glob 污染)。"""
    out = []
    for sb, sd in PAIRS:
        out.append({"kind": "ctrl", "seed": sb, "sb": sb, "sd": sd,
                    "mp4": OUT / f"ctrl_s{sb}.mp4"})
        out.append({"kind": "ctrl", "seed": sd, "sb": sb, "sd": sd,
                    "mp4": OUT / f"ctrl_s{sd}.mp4"})
        for band, lam in ARMS:
            out.append({"kind": "arm", "band": band, "lam": lam, "sb": sb, "sd": sd,
                        "mp4": OUT / f"{band}{int(round(lam*100)):02d}_s{sb}d{sd}.mp4"})
    return out


def build_manifest():
    return {
        "design": "T2V 域三模式对比: Wan2.2-TI2V-5B + 纯灰条件图(无信息 f0); "
                  "cow 两对 × field/ll/lp × λ{0.1,0.25,0.5,(1.0)}; 26 条",
        "why_gray_not_t2v_model": "Wan2.1-T2V-1.3B 换模型会混入模型变量; 灰图保持 同模型/"
                                   "同噪声种子/同算子/同判读, 只动条件信息量",
        "arms": [{"kind": j["kind"], "band": j.get("band"), "lam": j.get("lam"),
                  "pair": f"{j['sb']}↔{j['sd']}"} for j in jobs()],
        "metrics_prereg": {
            "M1": "同 GB 协议: iso 包络反解 λ* → raw 内插读 achieved; 命中=err≤30px; "
                  "field λ=1 锚 = donor ctrl(确定性恒等)",
            "M2": "reach=(朝 donor 可达端点极值−base)/sep; overshoot",
            "M3皮相": "跨视频 DINOv2 亲和: mean cos(臂主框crop, base ctrl crops) 与 "
                      "cos(臂, donor ctrl crops) 之差 gap = to_base − to_donor; "
                      "嵌合体判据: ll/lp@1.0 gap>0(皮相跟 base) 而 field(=donor ctrl) gap<0",
            "M3幻影": "实体读出同 ent 协议(灰图 f0 无检出, 无继承排除需求)"},
        "hypotheses": {
            "H_script": "三模式均可控剧本(剂量-响应存在), 方向朝 donor",
            "H_skin(核心)": "皮相归属: ll/lp@1.0 → base(嵌合体: donor 运动+base 皮相); "
                            "field@1.0 → donor; 证伪: ll/lp@1.0 皮相亲和 donor 或两头都不像",
            "H_viable": "灰图域可行性门: ctrl 主实体 ≥6/9 帧有效; 不过则如实报告域不可行"},
        "gates": "白性门逐臂; 帧0=灰图(算子内置不触); ctrl 可行性门",
    }


def ensure_manifest():
    p = HERE / "x0_manifest.json"
    if not p.exists():
        p.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=1))
        print(f"[prereg] {p} 落盘", flush=True)
    return p


def main():
    ensure_manifest()
    cond = ensure_cond()
    from PIL import Image
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    pipe = WanImageToVideoPipeline.from_pretrained(
        WAN22_MODEL_PATH,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.vae.enable_tiling()
    pipe.to("cuda")
    img = Image.open(cond)
    for j in jobs():
        if j["mp4"].exists() and j["mp4"].stat().st_size > 0:
            print(f"[skip] {j['mp4'].name}", flush=True)
            continue

        def noise(seed):
            g = torch.Generator(device="cpu").manual_seed(seed)
            return torch.randn((1, *SHAPE), generator=g)
        if j["kind"] == "ctrl":
            eps = noise(j["seed"])
        else:
            eps = blend_eps(noise(j["sb"])[0], noise(j["sd"])[0], j["lam"],
                            band=j["band"]).unsqueeze(0)
        g = whiteness_gate(eps[0, :, 1:])
        t0 = time.time()
        o = pipe(image=img, prompt=PROMPT, height=480, width=832, num_frames=81,
                 num_inference_steps=50, guidance_scale=5.0, latents=eps, output_type="np")
        export_to_video(o.frames[0], str(j["mp4"]), fps=16)
        del o
        torch.cuda.empty_cache()
        print(f"[{j['mp4'].stem}] 白性={g['pass']}(std {g['std']}) {time.time()-t0:.0f}s", flush=True)
    print("X0_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
