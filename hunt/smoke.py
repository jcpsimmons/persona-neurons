"""Smoke: our AddMLP steering reproduces L34 N3838 @0.75 behaviour; time a batched screen."""
import time, json
from common import *
model, tok = load_model()
cal_prompts = [p.strip() for p in (HERE.parent / "prompts.txt").read_text().splitlines() if p.strip() and not p.startswith("#")]
calib = calibrate(model, tok, cal_prompts)
say("calib L34", calib["34"], "(steer.py got 93.30)")
t = time.time()
base = gen(model, tok, cal_prompts, 80)
say(f"baseline batch {time.time()-t:.1f}s")
t = time.time()
with Steer(model, calib, [(34, 3838, 0.75)]):
    ev = gen(model, tok, cal_prompts, 80)
say(f"steered batch {time.time()-t:.1f}s")
for p, b, e in zip(cal_prompts, base, ev):
    print("==", p, "\n  BASE:", b[:200].replace("\n", " "), "\n  EVIL:", e[:200].replace("\n", " "))
