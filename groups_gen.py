"""groups_gen.py — GB/GA 两组对照实验生成器(2026-10-13, 预注册 groups_manifest.json)

GB 固定 f0: 既有配置下补 lp(rFFT 径向低通, frac=0.25=Haar LL1 同维度)臂,
   4 对 × λ{.25,.5,.75,1.0} = 16 条; field/ll 对照全部复用 Phase-1 既有臂。
GA 不固定 f0: donor 剧本改在 f0_B(aux seed ctrl 第 40 帧)条件下测量(4 条 donor-ctrl-f0B);
   λ 臂与 GB 逐位相同(同 cond=f0_A + 同混合噪声, 管线确定性)——两组差异**仅 donor 参照系**,
   GA 的 M1/M2 按 donor_end(f0_B) 重锚 ⇒ 无需重复生成臂。
预注册纪律: manifest 先于任何生成落盘; 幂等断点续跑; 逐场景子进程(OOM 加固同 phase1_gen)。
用法: python3 groups_gen.py            # 全流程(逐场景子进程: gb → ga)
      python3 groups_gen.py gb <scene> # 单场景单阶段(子进程入口)
      python3 groups_gen.py ga <scene>
TDD: tests/test_groups.py(臂表/命名防撞/manifest/帧提取)。
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from model_config import WAN22_MODEL_PATH
sys.path.insert(0, str(HERE))

from ll_blend import blend_eps
from ot_common import latent_shape_ot, whiteness_gate

MODEL = WAN22_MODEL_PATH
SHAPE = latent_shape_ot(81, 480, 832)
LP_FRAC = 0.25
GB_PAIRS = {"cow_p": [(9200, 9250), (9220, 9240)],
            "horse_f": [(9300, 9240)],
            "boat_l": [(9250, 9210)]}
GB_LAMS = (0.25, 0.5, 0.75, 1.0)
ORIG = {"cow_p": (9200, 9250), "dog_p": (9220, 9200),
        "horse_f": (9200, 9260), "boat_l": (9250, 9210)}
SCENES = {"cow_p": "a cow standing in a green pasture",
          "horse_f": "a horse galloping across an open green field",
          "boat_l": "a sailboat sailing across a calm blue lake"}
AUX_F0B = {"cow_p": 9230, "horse_f": 9220, "boat_l": 9240}   # f0_B 图像来源(不在任何对中)
F0B_FRAME = 40


def scene_dir(scene):
    return HERE / "lambda_scan" if scene == "cow_p" else HERE / "scenes" / scene


def cond_A(scene):
    return HERE / "payload/donor_cow_p_cond.png" if scene == "cow_p" else scene_dir(scene) / "cond.png"


def ctrl_path(scene, seed):
    if scene == "cow_p":
        return HERE / "videos" / f"cow_p_s{seed}_ctrlB.mp4"
    return scene_dir(scene) / f"{scene}_s{seed}_ctrlB.mp4"


def gb_arm_name(scene, sb, sd, band, lam):
    tag = f"i2v_lam_{band}{int(lam * 100):02d}_s{sb}"
    return tag + ("" if (sb, sd) == ORIG[scene] else f"d{sd}")


def gb_jobs():
    jobs = []
    for scene, pairs in GB_PAIRS.items():
        for sb, sd in pairs:
            for lam in GB_LAMS:
                jobs.append({"scene": scene, "sb": sb, "sd": sd, "band": "lp", "lam": lam,
                             "frac": LP_FRAC, "cond": cond_A(scene),
                             "mp4": scene_dir(scene) / f"{gb_arm_name(scene, sb, sd, 'lp', lam)}.mp4"})
    return jobs


def f0b_png(scene):
    return HERE / "groups" / "f0B" / f"{scene}.png"


def ga_donorctrl_path(scene, sd):
    return HERE / "groups" / "ga" / scene / f"{scene}_s{sd}_ctrlB_f0B.mp4"


def ga_jobs():
    seen, jobs = set(), []
    for scene, pairs in GB_PAIRS.items():
        for _, sd in pairs:
            mp4 = ga_donorctrl_path(scene, sd)
            if mp4 in seen:
                continue
            seen.add(mp4)
            jobs.append({"scene": scene, "seed": sd, "mp4": mp4, "cond": f0b_png(scene)})
    return jobs


def extract_frame(video, idx, out):
    """从视频取单帧存 png(f0_B 图像来源; 独立函数便于 TDD)。"""
    import imageio.v3 as iio
    from PIL import Image
    arr = np.asarray(iio.imread(str(video), index=idx))[..., :3]
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out)


def build_manifest():
    return {
        "design": "两组对照: GB=固定f0(补 lp 臂); GA=不固定f0 —— donor 剧本在 f0_B(aux seed ctrl "
                  "第40帧)条件下测量; λ 臂与 GB 逐位相同(管线确定性), 两组差异仅 donor 参照系, "
                  "GA 的 M1/M2 按 donor_end(f0_B) 重锚",
        "GB": {"pairs": {k: [list(p) for p in v] for k, v in GB_PAIRS.items()},
               "bands": [["lp", l] for l in GB_LAMS], "frac": LP_FRAC,
               "note": "lp=rFFT 径向低通(实 dof 累计, 自共轭列计1其余计2); frac=0.25 ⇒ 390±1 实dof/帧 "
                       "= Haar LL1(15×26)同维度; 命名沿 phase1 规则(原对无后缀, 其余 d{donor})"},
        "GA": {"aux_f0b": AUX_F0B, "f0b_frame": F0B_FRAME,
               "donors": sorted(f"{s}_s{d}" for s, d in {(j["scene"], j["seed"]) for j in ga_jobs()}),
               "note": "新生成仅 donor-ctrl-f0B(4条); λ=1 field 臂 ≡ donor seed 在 f0_A 下的 ctrl"
                       "(池内已有, 确定性); aux seed 不在任何对中(防谱系污染)"},
        "metrics_prereg": {
            "M1": "位置校准/命中: target=base+frac·sep(frac∈{.25,.5,.75}); 该模式该对 iso(PAVA)包络"
                  "首达反解 λ*; achieved=raw 线性内插; err=|achieved−target|; 命中=err≤30px; "
                  "双口径(逐对上界 / LOO 总体)",
            "M2": "动态范围: reach=(朝 donor 方向可达端点极值−base)/sep; overshoot=max(0,reach−1); "
                  "GA 组 sep 用 donor 端点(f0_B)重算",
            "M3": "主体一致性: 主框 crop(DINOv2-small 嵌入)对自身 f0 的 cos 均值(t≥1)+斜率; "
                  "幻影率=存在与主框 IoU<0.2 可信他框的帧占比(可信规则同判读 v4: boat 船签名, "
                  "其余场景 conf≥0.15)"},
        "hypotheses": {
            "H1": "field(白噪声整场复制)在 M1/M2 优于 lp≈ll(控制力最强); 证伪: lp/ll 命中率或 reach "
                  "系统性高于 field",
            "H2": "lp≈ll 在 M3 主体一致性优于 field 且幻影率更低(低频复制保留 base 细节带); "
                  "证伪: field 一致性 ≥ lp/ll 或幻影率无差",
            "GA_pred": "剧本地址条件化: donor(f0_B) 参照下各模式 reach 显著低于 GB; 若 reach≈1 则剧本"
                       "跨条件可移植(两结果均有信息量)"},
        "gates": "白性门(std±0.01)逐臂打印; 帧0不触(算子内置); GB 锚: field100≡donor ctrl(Phase1 已证 "
                 "4/4); GA donor-ctrl-f0B 判读后与 f0_A 池端点对照",
        "n_videos": len(gb_jobs()) + len(ga_jobs()),
    }


def ensure_manifest():
    p = HERE / "groups_manifest.json"
    if not p.exists():
        p.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=1))
        print(f"[prereg] {p} 落盘", flush=True)
    return p


def sample_noise(shape, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g)


def load_pipe():
    from diffusers import WanImageToVideoPipeline
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.vae.enable_tiling()
    pipe.to("cuda")
    return pipe


def gen(pipe, mp4, **kw):
    from diffusers.utils import export_to_video
    mp4.parent.mkdir(parents=True, exist_ok=True)   # groups/ga/<scene>/ 等输出目录(2026-10-13 事故)
    t0 = time.time()
    o = pipe(height=480, width=832, num_frames=81, num_inference_steps=50,
             guidance_scale=5.0, output_type="np", **kw)
    export_to_video(o.frames[0], str(mp4), fps=16)
    del o
    torch.cuda.empty_cache()
    return time.time() - t0


def stage_gb(scene=None):
    from PIL import Image
    pipe = None
    for j in gb_jobs():
        if scene is not None and j["scene"] != scene:
            continue
        if j["mp4"].exists() and j["mp4"].stat().st_size > 0:
            print(f"[skip] gb {j['mp4'].name}", flush=True)
            continue
        if pipe is None:
            pipe = load_pipe()
        eb = sample_noise((1, *SHAPE), seed=j["sb"])[0]
        ed = sample_noise((1, *SHAPE), seed=j["sd"])[0]
        eps = blend_eps(eb, ed, j["lam"], band="lp", frac=j["frac"]).unsqueeze(0)
        g = whiteness_gate(eps[0, :, 1:])
        dt = gen(pipe, j["mp4"], image=Image.open(j["cond"]),
                 prompt=SCENES[j["scene"]], latents=eps)
        print(f"[gb {j['scene']} s{j['sb']}←s{j['sd']} lp{j['lam']}] "
              f"白性={g['pass']}(std {g['std']}) {dt:.0f}s", flush=True)


def stage_ga(scene=None):
    from PIL import Image
    scenes = [scene] if scene else list(AUX_F0B)
    for s in scenes:                                  # f0_B 图像幂等提取
        if not f0b_png(s).exists():
            extract_frame(ctrl_path(s, AUX_F0B[s]), F0B_FRAME, f0b_png(s))
            print(f"[f0B] {s} ← {s}_s{AUX_F0B[s]}_ctrlB f{F0B_FRAME}", flush=True)
    pipe = None
    for j in ga_jobs():
        if scene is not None and j["scene"] != scene:
            continue
        if j["mp4"].exists() and j["mp4"].stat().st_size > 0:
            print(f"[skip] ga {j['mp4'].name}", flush=True)
            continue
        if pipe is None:
            pipe = load_pipe()
        eps = sample_noise((1, *SHAPE), seed=j["seed"])
        g = whiteness_gate(eps[0, :, 1:])
        dt = gen(pipe, j["mp4"], image=Image.open(j["cond"]),
                 prompt=SCENES[j["scene"]], latents=eps)
        print(f"[ga {j['scene']} donor s{j['seed']}@f0B] "
              f"白性={g['pass']}(std {g['std']}) {dt:.0f}s", flush=True)


def main():
    ensure_manifest()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for scene in ["horse_f", "boat_l", "cow_p"]:     # 判读不友好场景先行
        for stage in ("gb", "ga"):
            print(f"=== {stage} {scene} (subprocess) ===", flush=True)
            r = subprocess.run([sys.executable, __file__, stage, scene], env=env)
            if r.returncode != 0:
                sys.exit(r.returncode)
    print("GROUPS_GEN_DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] in ("gb", "ga"):
        (stage_gb if sys.argv[1] == "gb" else stage_ga)(sys.argv[2])
    else:
        main()
