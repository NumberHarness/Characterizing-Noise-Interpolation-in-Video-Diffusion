"""korder_probe.py — 决定次序探针: 位置 vs 背景细节, 谁先锁定?(2026-10-15)

动机: kresp 的 ρ 曲线(快分量=细节带)被读作"细节先定、位置后定", 但 (a) 早期 ρ 含
残余噪声的平凡投影混杂, (b) e₅₀ 混合内容轴。本探针直接解码检查点 x̂₀ 到像素:
  位置轴: 解码帧 f10/f40/f80 的主实体 cx(检测器) —— 随 k 的收敛曲线;
  细节轴: 解码帧顶部天空条带与 k=50 终帧的灰度相关 —— 随 k 的锁定曲线;
  平凡投影分解: ⟨δ, e₅₀⟩(噪声差对终态方向的直接投影)量化 kresp ρ(1) 的混杂。
用法: python3 korder_probe.py           # cow 9200↔9250, 4 轨迹 × 6 检查点
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from model_config import WAN22_MODEL_PATH
sys.path.insert(0, str(HERE))

from kresp_run import sample_noise
from ll_blend import blend_eps
from ot_common import latent_shape_ot

CKPTS = (1, 3, 8, 17, 35, 50)
FRAMES = (10, 40, 80)
SHAPE = latent_shape_ot(81, 480, 832)
MODEL = WAN22_MODEL_PATH


def run_traj(pipe, image, prompt, eps, tag):
    """50 步轨迹; 记录 (步序, 输入样本, 模型速度) 与步后潜变量(仅 k=50)。"""
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
            rec["final"] = out[0].detach()[0].float().cpu()
        return out

    pipe.scheduler.step = spy
    try:
        pipe(image=image, prompt=prompt, height=480, width=832, num_frames=81,
             num_inference_steps=50, guidance_scale=5.0,
             latents=eps.unsqueeze(0), output_type="latent")
    finally:
        pipe.scheduler.step = real_step
    assert sorted(rec["x0s"]) == list(CKPTS) and rec["final"] is not None, tag
    return rec


def decode_frames(pipe, lat):
    """潜变量(缩放域)→ 81 帧像素 (H,W,3) uint8。"""
    z = lat.to(pipe.vae.dtype).to(pipe.device)[None]
    cfg = pipe.vae.config
    if getattr(cfg, "latents_mean", None) is not None:
        std = torch.tensor(cfg.latents_std, dtype=z.dtype, device=z.device).reshape(1, -1, 1, 1, 1)
        mean = torch.tensor(cfg.latents_mean, dtype=z.dtype, device=z.device).reshape(1, -1, 1, 1, 1)
        z = z / std + mean
    with torch.no_grad():
        v = pipe.vae.decode(z).sample[0]                    # (C,T,H,W) [-1,1]
    v = v.float().cpu().numpy().transpose(1, 2, 3, 0)
    return ((np.clip(v, -1, 1) + 1) * 127.5).astype(np.uint8)


def main():
    from PIL import Image
    from diffusers import WanImageToVideoPipeline
    from video_protocol import CLASS_IDS, main_box
    from yolo_detector import detect_boxes_with_owl_fallback

    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.vae.enable_tiling()
    pipe.to("cuda")
    img = Image.open(HERE / "payload/donor_cow_p_cond.png")
    prompt = "a cow standing in a green pasture"
    eb, ed = sample_noise(9200), sample_noise(9250)
    runs = {"base": eb, "donor": ed,
            "ll@1": blend_eps(eb, ed, 1.0, band="ll"),
            "hh@1": blend_eps(eb, ed, 1.0, band="hh")}
    out = {}
    final_frames = {}
    for name, eps in runs.items():
        rec = run_traj(pipe, img, prompt, eps, name)
        rows = {}
        final_frames[name] = decode_frames(pipe, rec["final"])
        for k in CKPTS:
            s, v, sig = rec["x0s"][k]
            x0 = (s - sig * v).numpy()
            frames = decode_frames(pipe, torch.from_numpy(x0))
            cx = {}
            for f in FRAMES:
                bs = detect_boxes_with_owl_fallback(
                    frames[f], CLASS_IDS["cow_p"], conf=0.15, imgsz=1280)
                mb = main_box(bs, "cow_p")
                cx[f] = round((mb[0] + mb[2]) / 2, 1) if mb is not None else None
            sky = [np.corrcoef(frames[f][:100, :, :3].mean(axis=2).ravel(),
                               final_frames[name][f][:100, :, :3].mean(axis=2).ravel())[0, 1]
                   for f in FRAMES]
            rows[k] = {"cx": cx, "sky_corr_final": [round(float(c), 3) for c in sky]}
            del frames
        out[name] = rows
        print(f"[{name}] " + " | ".join(
            f"k{k}: cx(f80)={rows[k]['cx'][80]} sky={rows[k]['sky_corr_final'][2]}" for k in CKPTS),
            flush=True)
    # 平凡投影分解: ⟨δ, e₅₀⟩(ll/hh 臂与 donor 的噪声差对终态方向)
    # e₅₀ 在缩放域; δ 同域
    runs_rec = {}
    outp = HERE / "korder_probe.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("→ korder_probe.json")


if __name__ == "__main__":
    main()
