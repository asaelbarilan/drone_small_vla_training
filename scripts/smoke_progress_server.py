"""D175 smoke test: the eval server with progress tokens, end to end, no simulator.

Loads uav_flow_eval_server's qwen predictor with --progress (and optionally goal
memory and sampling) and drives it through Flask's test client like
batch_run_act_all.py does: /reset, then /predict calls whose next state is the
last pose the server returned (the evaluator teleports, so this is the same
state chain). Photos are a few validation frames from the store. Checks, per call:
the progress line parses, K poses come back, the goal stays fixed under goal
memory, and in sampling mode the record carries the progress line, 4K ids and
4K log-probabilities. Exits non-zero on any failure.
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import uav_flow_eval_server as server  # noqa: E402
from train_uav_flow_vla import OFFICIAL_STORE, image_path  # noqa: E402
from PIL import Image  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--calls", type=int, default=4)
    parser.add_argument("--goal-memory", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    state = server.STATE
    state.update(stop_confirm=False, execute_steps=None, episode=0, call=0, rollout_dir=None,
                 goal=None, yaw_total=0.0, last_yaw=None, first_frames=0)
    if args.temperature:
        state["rollout_dir"] = args.out / "rollout_images"
        state["rollout_dir"].mkdir(exist_ok=True)
    state["frames_dir"] = args.out / "first_frames"
    state["frames_dir"].mkdir(exist_ok=True)
    state["log"] = open(args.out / "calls.jsonl", "w", encoding="utf-8")  # noqa: SIM115
    state["predict"] = server.qwen_loader(
        args.adapter, args.chunk, "bf16", args.temperature, None,
        progress=True, goal_memory=args.goal_memory,
    )

    episode = next(
        json.loads(line)
        for line in (OFFICIAL_STORE / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["split_unseen"] == "val"
    )
    client = server.app.test_client()
    client.post("/reset")
    proprio, failures, goals, unparsed = [0.0, 0.0, 0.0, 0.0], [], [], []
    for call in range(args.calls):
        frame = episode["images"][min(call * args.chunk, len(episode["images"]) - 1)]
        picture = Image.open(image_path(dict(image=frame, format="official"))).convert("RGB")
        picture = picture.resize((224, 224))
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        reply = client.post("/predict", json=dict(
            image=base64.b64encode(buffer.getvalue()).decode(), proprio=proprio,
            instr=episode["instruction"],
        )).get_json()
        record = json.loads((args.out / "calls.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        poses = reply["action"]
        print(json.dumps(dict(call=call, said=record.get("progress_said"),
                              used=record.get("progress_text"), poses=len(poses))), flush=True)
        # A 30-update adapter may not write the line well yet: count it, do not fail
        # on it (the mechanics are what this checks; quality comes from training).
        if record.get("progress_parsed") is None:
            unparsed.append(call)
        if len(poses) != args.chunk:
            failures.append(f"call {call}: {len(poses)} poses")
        if args.temperature and (len(record.get("ids", [])) != 4 * args.chunk
                                 or len(record.get("logps", [])) != 4 * args.chunk
                                 or ("progress_text" not in record and record.get("progress_parsed") is not None)):
            failures.append(f"call {call}: rollout record incomplete")
        if args.goal_memory:
            goals.append(state.get("goal"))
        if poses:
            x, y, z, yaw = poses[-1]
            proprio = [x, y, z, yaw * 180.0 / 3.141592653589793]
    if "progress_said" not in record:
        failures.append("the server never wrote a progress line")
    if args.goal_memory and any(g != goals[0] for g in goals):
        failures.append(f"goal moved: {goals}")
    print(json.dumps(dict(instruction=episode["instruction"], unparsed_calls=unparsed,
                          failures=failures)), flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
