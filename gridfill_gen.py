"""gridfill_gen.py — λ 网格统一补齐(2026-10-16, 预注册 gridfill_manifest.json)

两个刹车案例统一到完整网格 {0.1, 0.25, 0.5, 0.75, 0.9, 1.0}(field@1.0 ≡ donor ctrl):
  horse 9300↔9240 (I2V): ll@{0.1,0.75,0.9} + lp@{0.1,0.9}          —— 5 条
  gray cow 9220↔9240:    field/ll/lp@{0.75,0.9}                     —— 6 条
P-g1: horse ll 翻转点 λ* ∈ (0.5,1](ll@0.5=275 钉, ll@1.0=433 翻);
P-g2: gray field 翻转点 ∈ (0.5,1); ll/lp 的 0.75/0.9 给出回落形态(671→608 单调与否)。
"""
import json, sys, time
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ll_blend import blend_eps
from ot_common import latent_shape_ot, whiteness_gate
from groups_gen import gen, load_pipe

SHAPE = latent_shape_ot(81, 480, 832)
HORSE = ("a horse galloping across an open green field", HERE / "scenes/horse_f/cond.png")
GRAY = ("a cow standing in a green pasture", HERE / "x0/cond_gray.png")
ARMS = ([("ll", l, "horse", HORSE, HERE/f"scenes/horse_f/i2v_lam_ll{int(l*100):02d}_s9300d9240.mp4")
         for l in (0.1, 0.75, 0.9)] +
        [("lp", l, "horse", HORSE, HERE/f"scenes/horse_f/i2v_lam_lp{int(l*100):02d}_s9300d9240.mp4")
         for l in (0.1, 0.9)] +
        [("field", l, "gray", GRAY, HERE/f"x0/field{int(l*100):02d}_s9220d9240.mp4") for l in (0.75, 0.9)] +
        [("ll", l, "gray", GRAY, HERE/f"x0/ll{int(l*100):02d}_s9220d9240.mp4") for l in (0.75, 0.9)] +
        [("lp", l, "gray", GRAY, HERE/f"x0/lp{int(l*100):02d}_s9220d9240.mp4") for l in (0.75, 0.9)])

def jobs():
    return [{"band": b, "lam": l, "mp4": mp4, "cond": cond, "prompt": prompt}
            for b, l, tag, (prompt, cond), mp4 in ARMS]

def ensure_manifest():
    p = HERE / "gridfill_manifest.json"
    if not p.exists():
        p.write_text(json.dumps({
            "design": "λ 网格统一: 两刹车案例补至 {0.1,0.25,0.5,0.75,0.9,1.0}",
            "arms": [{"band": b, "lam": l} for b, l, *_ in ARMS],
            "predictions": {"P-g1": "horse ll 翻转点 ∈ (0.5,1]; ll@0.1 预计仍钉 base",
                            "P-g2": "gray field 翻转点 ∈ (0.5,1); ll/lp 回落形态待测"},
            "gates": "白性门逐臂; 帧0不触"}, ensure_ascii=False, indent=1))
        print(f"[prereg] {p} 落盘", flush=True)
    return p

def main():
    ensure_manifest()
    from PIL import Image
    pipe = None
    for j in jobs():
        if j["mp4"].exists() and j["mp4"].stat().st_size > 0:
            print(f"[skip] {j['mp4'].name}", flush=True); continue
        if pipe is None:
            pipe = load_pipe()
        eb = torch.randn((1, *SHAPE), generator=torch.Generator().manual_seed(9300 if j["band"] != "field" or j["mp4"].parent.name == "x0" else 9300))[0]
        # 网格对应的种子: horse 对 9300↔9240; gray 对 9220↔9240
        sb = 9300 if j["mp4"].parent.name != "x0" else 9220
        eb = torch.randn((1, *SHAPE), generator=torch.Generator().manual_seed(sb))[0]
        ed = torch.randn((1, *SHAPE), generator=torch.Generator().manual_seed(9240))[0]
        eps = blend_eps(eb, ed, j["lam"], band=j["band"]).unsqueeze(0)
        g = whiteness_gate(eps[0, :, 1:])
        dt = gen(pipe, j["mp4"], image=Image.open(j["cond"]), prompt=j["prompt"], latents=eps)
        print(f"[{j['mp4'].name}] 白性={g['pass']}(std {g['std']}) {dt:.0f}s", flush=True)
    print("GRIDFILL_DONE", flush=True)

if __name__ == "__main__":
    main()
