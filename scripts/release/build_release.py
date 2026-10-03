"""Release package for the Qwen3-VL-4B UAV-Flow VLA (D183).

Writes, into the release folder (default D:/drone_vla_pilot/release/qwen3vl4b_uavflow_vla_progress):
  training/phases.json   every training phase: data, init, schedule, updates, hardware, outcome
  training/curves.png    training loss, held-out move-token accuracy (real / simulator), learning rate
  eval/results.json      flight-test table (success 3 m / 10 deg, nDTW, per class) for every variant
The weights, logs, scores and data lists are copied in by hand (see the README); this script
only derives the summary files from them.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

PHASES = [
    dict(name="phase0_real10", decision="D159/D160", init="Qwen/Qwen3-VL-4B-Instruct (bf16, no adapter)",
         data="10 real UAV-Flow shards: 1,987 train flights (site split, instruction sweep), mirror copies, both wordings",
         schedule="cosine, peak lr 5e-4, warmup 3%", updates="1-2,500", batch="8 x accum 4 = 32",
         outcome="adapter_s2500; closed loop nDTW 0.128 / success 38% (section 1, 10 of the report)"),
    dict(name="phase1_real_plus_sim", decision="D165", init="phase0 adapter_s2500",
         data="+ UAV-Flow-Sim 9,941 flights (168 near-test flights excluded: data/uav_flow_sim_excluded.json)",
         schedule="cosine, peak lr 2e-4", updates="1-2,000", batch="32",
         outcome="adapter_s2000; nDTW 0.336 / success 50%"),
    dict(name="phase2_low_lr", decision="D169b", init="phase1 adapter_s2000",
         data="same; 504 simulator flights held out for validation (56 per motion type, data/sim_val_flights.json)",
         schedule="cosine, peak lr 5e-5, early stop on simulator loss (patience 3)", updates="1-about 2,700 (best 2,500)",
         batch="32", outcome="adapter_best (update 2,500); not flight-tested"),
    dict(name="phase3_all_shards", decision="D170/D171", init="phase2 adapter_best",
         data="all 54 real shards (+21,795 flights: 18,430 train, 3,365 val; no instruction sweep on the new shards) + simulator",
         schedule="cosine, peak lr 5e-4, warmup 3% of a planned 26,000", updates="1-18,000", batch="32",
         outcome="adapter_s18000: nDTW 0.367 / success 52% (no progress line)"),
    dict(name="phase4_progress_steady", decision="D176/D177", init="phase3 checkpoint 18,000 (optimizer state resumed)",
         data="same + progress tokens (the 'Left' line before the moves)",
         schedule="warmup-stable-decay: steady lr 1.142e-4 (the cosine value at 18,000, no new warmup); plateau rule on "
                  "real accuracy (D176), real + simulator accuracy from 26,000 (D177)",
         updates="18,001-28,000", batch="32", outcome="checkpoints every 2,000"),
    dict(name="phase5_decay", decision="D179", init="checkpoint 28,000 (resumed)",
         data="same, progress tokens on", schedule="cosine 1.142e-4 -> 0 over 3,000 updates", updates="28,001-31,000",
         batch="32", outcome="adapter_s31000 (FINAL): line 61%; line + memory + deadband 69% / nDTW 0.449"),
]
COMMON = dict(
    base_model="Qwen/Qwen3-VL-4B-Instruct", precision="bf16 (unquantised base)",
    lora=dict(rank=32, alpha=32, dropout=0.0, target_modules="all-linear (vision tower, projector and language model)"),
    actions="4-D drone-frame (dx, dy, dz, dyaw) per step, K = 8 steps per answer, 256 bins per value on the last 256 "
            "vocabulary ids, range = 0.1/99.9 percentiles (data/action_stats_manifest_10shard.json)",
    prompt="Current State: {x,y,z,yaw_deg}, What action should the uav take to {instruction}?  (+ one 224x224 photo)",
    progress_line="'Left +FF.F,+SS.S,+UU.U,+YYY' = metres forward, sideways, up and degrees of turn left to the "
                  "flight's end, in the drone frame; 21 tokens, written before the 32 move tokens (phases 4-5)",
    hardware="1x NVIDIA A10G 24 GB (AWS g5.2xlarge), about 6.2-6.8 s per update of 32 examples",
    total_updates="about 38,700 across all phases (2,500 + 2,000 + about 2,700 + 31,000)",
)


def lr_at(step):
    if step <= 18000:
        warm, total = round(0.03 * 26000), 26000
        if step <= warm:
            return 5e-4 * step / warm
        return 5e-4 * 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))
    if step <= 28000:
        return 1.142e-4
    return 1.142e-4 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - 28000) / 3000)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("D:/drone_vla_pilot/release/qwen3vl4b_uavflow_vla_progress"))
    args = parser.parse_args()
    r = args.release
    (r / "training" / "phases.json").write_text(json.dumps(dict(common=COMMON, phases=PHASES), indent=2), encoding="utf-8")

    report = json.loads((r / "training" / "report_long_run.json").read_text(encoding="utf-8"))
    losses = np.asarray(report["losses"], dtype=float)
    held = report["held_out"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    steps = np.arange(1, len(losses) + 1)
    window = 200
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    axes[0].plot(steps[window - 1:], smooth, lw=1)
    axes[0].set_ylabel("train loss (200-update mean)")
    axes[0].set_title("Long run (phases 3-5): progress line on at 18,000, decay from 28,000")
    s = [e["step"] for e in held]
    axes[1].plot(s, [e["token_acc"] for e in held], marker="o", ms=3, label="real flights, exact")
    axes[1].plot(s, [e.get("sim_token_acc") for e in held], marker="o", ms=3, label="simulator, exact")
    axes[1].set_ylabel("held-out move-token accuracy")
    axes[1].legend()
    axes[2].plot(steps, [lr_at(x) for x in steps], lw=1)
    axes[2].set_ylabel("learning rate")
    axes[2].set_xlabel("update")
    for ax in axes:
        for x in (18000, 28000):
            ax.axvline(x, color="grey", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(r / "training" / "curves.png", dpi=130)

    results = {}
    for name in sorted(p.name for p in (r / "eval").iterdir() if p.is_dir()):
        flights = json.loads((r / "eval" / name / "score.json").read_text(encoding="utf-8"))["flights"]
        per = {}
        for f in flights:
            per.setdefault(f["cls"], []).append(f["end_distance_m"] <= 3 and f["final_yaw_error_deg"] <= 10)
        ndtw = [f["ndtw"] for f in flights if f["ndtw"] is not None]
        results[name] = dict(flights=len(flights), success=sum(sum(v) for v in per.values()) / len(flights),
                             mean_ndtw=round(float(np.mean(ndtw)), 4),
                             per_class_success={k: f"{sum(v)}/{len(v)}" for k, v in per.items()})
    (r / "eval" / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, v in results.items():
        print(f"{name:24s} success {v['success']:.2f}  nDTW {v['mean_ndtw']}")


if __name__ == "__main__":
    main()
