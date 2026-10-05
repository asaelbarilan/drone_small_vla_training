"""D162: model server for UAV-Flow-Eval (UnrealZoo DowntownWest, closed loop).

Speaks the official protocol of OpenVLA-UAV/vla-scripts/openvla_act.py so the
unmodified batch_run_act_all.py drives it:

  POST /predict {image: base64 PNG 224x224, proprio: [x, y, z, yaw_deg], instr}
  -> {action: [[x, y, z, yaw_rad], ...]}   poses in the START frame, sim units
  POST /reset

Two models:

  openvla-uav  the released checkpoint, exactly the official server's maths
               (predict_action with unnorm_key "sim", rotate the step by the
               current yaw, add the current position and yaw). Deviation: NF4 on
               an 8 GB card instead of bf16 + flash-attention, plus the D156
               attention-mask fix for the appended 29871 token.
  qwen         our adapter (official format, K-step chunks, trained on REAL
               flights in metres). The simulator works in centimetres - the
               released model's "sim" action stats are ~10-50 per step - so the
               state is divided by 100 before it goes into our prompt and every
               predicted step is multiplied by 100. Each of the K steps is
               integrated in turn in the drone frame and returned as K poses,
               which the evaluator executes one after another.
               --stop-confirm: an all-still chunk returns only its first pose.
"""

import argparse
import base64
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)
STATE = {}
# The evaluator ends a task after 10 consecutive moves below these (sim units,
# degrees). With K-step chunks one all-still answer is K of those 10 at once, so
# --stop-confirm executes one still move and asks again instead (D168); a flight
# then ends only when the model keeps answering "stop", as a one-step model must.
STILL_POS, STILL_YAW_DEG = 3.0, 1.0


def openvla_loader(path, precision):
    sys.path.insert(0, "D:/drone_vla_pilot/models/OpenFly-Platform/train")
    import timm

    original = timm.create_model
    timm.create_model = lambda *a, **k: original(*a, **{**k, "pretrained": False})
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        quantization_config=None
        if precision == "bf16"
        else BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["vision_backbone", "projector", "lm_head"],
        ),
        device_map={"": 0},
        attn_implementation="eager",
    )
    model.eval()
    timm.create_model = original

    def predict(image, proprio, instruction):
        state = ",".join(str(round(float(x), 1)) for x in proprio)
        prompt = (
            f"In: Current State: {state}, What action should the uav take to {instruction}?\nOut:"
        )
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        if inputs["input_ids"][0, -1].item() != 29871:
            inputs["input_ids"] = torch.cat(
                [inputs["input_ids"], inputs["input_ids"].new_tensor([[29871]])], dim=1
            )
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], inputs["attention_mask"].new_ones((1, 1))], dim=1
            )
        with torch.inference_mode():
            step = model.predict_action(**inputs, unnorm_key="sim", do_sample=False)
        return [np.asarray(step, dtype=float)]  # one step, sim units

    return predict


def goal_from_left(left, position_m, yaw_total_deg):
    """D175 goal memory: a progress line (drone frame at this pose) -> the goal in
    the start frame (metres, summed yaw in degrees)."""
    yaw = math.radians(yaw_total_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return [
        position_m[0] + c * left[0] - s * left[1],
        position_m[1] + s * left[0] + c * left[1],
        position_m[2] + left[2],
        yaw_total_deg + left[3],
    ]


def left_from_goal(goal, position_m, yaw_total_deg):
    """Inverse of goal_from_left: what is left to a remembered goal from here."""
    yaw = math.radians(yaw_total_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = goal[0] - position_m[0], goal[1] - position_m[1]
    return [c * dx + s * dy, -s * dx + c * dy, goal[2] - position_m[2], goal[3] - yaw_total_deg]


def qwen_loader(
    adapter, chunk, precision, temperature=None, contrast=None, plausible=0.1,
    progress=False, goal_memory=False, deadband=None, constrain=True, box=False,
):
    from peft import PeftModel
    from train_uav_flow_vla import (
        OFFICIAL_PROMPT,
        append_ids,
        box_text,
        collate_left,
        encode,
        official_stats,
        parse_box,
        parse_progress,
        progress_text,
        setup,
    )
    from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList
    from uav_flow_action_tokenizer import ActionTokenizer

    api, processor, base, _ = setup("qwen", precision)
    tokenizer = ActionTokenizer(processor.tokenizer, official_stats())
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    low, high = tokenizer.vocab_size - tokenizer.n_bins, tokenizer.vocab_size

    class ActionTokensOnly(LogitsProcessor):
        def __call__(self, input_ids, scores):
            masked = torch.full_like(scores, float("-inf"))
            masked[:, low:high] = scores[:, low:high]
            return masked

    # D172 rollout sampling: only the 256 action tokens, at `temperature`, with
    # every other sampling tweak (top-k, top-p, repetition penalty) switched off,
    # so the recorded log-probabilities are exactly the policy RL updates:
    # log_softmax(logits[action tokens] / temperature).
    sampling = GenerationConfig(
        do_sample=True,
        temperature=temperature or 1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_new_tokens=4 * chunk,
        min_new_tokens=4 * chunk,
        output_logits=True,
        return_dict_in_generate=True,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    class ContrastWithGray(LogitsProcessor):
        """D173 contrastive decoding (PCD, 2505.13255; VCD): row 0 sees the real
        photo, row 1 a flat gray one. Both rows get the same contrasted scores
        (1 + a) * real - a * gray over the action tokens, restricted to tokens the
        real row finds plausible (p >= plausible * max p), so greedy picks the
        same token for both and the answer leans on what the camera adds."""

        def __call__(self, input_ids, scores):
            real, gray = scores[0, low:high].float(), scores[1, low:high].float()
            mixed = (1 + contrast) * real - contrast * gray
            probs = real.softmax(-1)
            mixed[probs < plausible * probs.max()] = float("-inf")
            out = torch.full_like(scores, float("-inf"))
            out[:, low:high] = mixed.to(scores.dtype)
            return out

    line_tokens = len(processor.tokenizer.encode(progress_text([0, 0, 0, 0]), add_special_tokens=False))

    def progress_line(row, image, metres):
        """D175 phase 1: the model writes its progress line (greedy). With goal
        memory, the first well-formed line fixes a goal in the start frame, and
        every later call is handed the line recomputed from that goal instead."""
        batch = encode(processor, tokenizer, row, chunk, with_answer=False, image=image)
        with torch.inference_mode():
            ids = model.generate(
                **api.cuda(batch), max_new_tokens=line_tokens, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )[0, batch["input_ids"].shape[1] :].tolist()
        said = processor.tokenizer.decode(ids, skip_special_tokens=True)
        left = parse_progress(said)
        used = left
        if goal_memory:
            if STATE.get("goal") is None and left is not None:
                STATE["goal"] = goal_from_left(left, metres, STATE["yaw_total"])
            if STATE.get("goal") is not None:
                used = left_from_goal(STATE["goal"], metres, STATE["yaw_total"])
                # D182: the model only ever learned lines that count down to exactly 0;
                # a small remainder or an overshoot ("-0.2 m", "-2 deg") kept it moving.
                # Inside the deadband the goal counts as reached: hand it the 0 line.
                if deadband and math.hypot(*used[:3]) < deadband[0] and abs(used[3]) < deadband[1]:
                    used = [0.0, 0.0, 0.0, 0.0]
        info = dict(progress_said=said, progress_parsed=left)
        if used is None:
            return None, info
        text = progress_text(used)
        return text, info | dict(progress_text=text)

    box_prefix = processor.tokenizer.encode('{"bbox_2d": [', add_special_tokens=False)

    def box_line(row, image):
        """D193 phase 0: the model first writes the box of the instruction's object (greedy,
        after a forced '{"bbox_2d": [' prefix); the line is then context for the progress
        line and the moves, exactly as in box-first training."""
        batch = append_ids(encode(processor, tokenizer, row, chunk, with_answer=False, image=image), box_prefix)
        with torch.inference_mode():
            ids = model.generate(
                **api.cuda(batch), max_new_tokens=24, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )[0, batch["input_ids"].shape[1] :].tolist()
        found = parse_box('{"bbox_2d": [' + processor.tokenizer.decode(ids, skip_special_tokens=True))
        return box_text(found or []), found

    def predict(image, proprio, instruction):
        metres = [proprio[0] / 100, proprio[1] / 100, proprio[2] / 100, proprio[3]]
        state = ",".join(str(round(float(x), 1)) for x in metres)
        prompt = OFFICIAL_PROMPT.format(state=state, instruction=instruction)
        row = dict(prompt=prompt)
        if box:
            row["box_text"], STATE["box"] = box_line(row, image)
        if progress:
            text, info = progress_line(row, image, metres)
            STATE["progress"] = info
            if text is not None:  # malformed line: answer without one
                row["progress_text"] = text
        batch = encode(processor, tokenizer, row, chunk, with_answer=False, image=image)
        with torch.inference_mode():
            if contrast:
                gray = Image.new("RGB", image.size, (127, 127, 127))
                pair = collate_left(
                    [batch, encode(processor, tokenizer, row, chunk, with_answer=False, image=gray)],
                    processor.tokenizer.pad_token_id,
                )
                ids = model.generate(
                    **api.cuda(pair),
                    max_new_tokens=4 * chunk,
                    min_new_tokens=4 * chunk,
                    do_sample=False,
                    logits_processor=LogitsProcessorList([ContrastWithGray()]),
                    pad_token_id=processor.tokenizer.pad_token_id,
                )[0, pair["input_ids"].shape[1] :].tolist()
            elif temperature:
                out = model.generate(
                    **api.cuda(batch),
                    generation_config=sampling,
                    logits_processor=LogitsProcessorList([ActionTokensOnly()]),
                )
                ids = out.sequences[0, batch["input_ids"].shape[1] :].tolist()
                logits = torch.stack(out.logits, dim=1)[0, :, low:high].float() / temperature
                picked = torch.tensor(ids, device=logits.device) - low
                logps = logits.log_softmax(-1).gather(1, picked[:, None])[:, 0].tolist()
                STATE["sample"] = dict(prompt=prompt, ids=ids, logps=logps, temperature=temperature)
                if "progress_text" in row:
                    STATE["sample"]["progress_text"] = row["progress_text"]
            elif constrain:
                # D191: greedy over the 256 action tokens only, exactly 4 x chunk of them. Before,
                # greedy was unconstrained and non-action tokens were dropped afterwards; on
                # "Orbit the dog clockwise" the model put a non-action token in every yaw slot,
                # 24 of 32 were left, and the server answered no moves (flight ended at once).
                # Where the best token already is an action token the answer is unchanged.
                ids = model.generate(
                    **api.cuda(batch), max_new_tokens=4 * chunk, min_new_tokens=4 * chunk, do_sample=False,
                    use_cache=True, logits_processor=LogitsProcessorList([ActionTokensOnly()]),
                    pad_token_id=processor.tokenizer.pad_token_id,
                )[0, batch["input_ids"].shape[1] :].tolist()
            else:
                ids = model.generate(
                    **api.cuda(batch), max_new_tokens=4 * chunk + 2, do_sample=False, use_cache=True
                )[0, batch["input_ids"].shape[1] :].tolist()
                if progress:
                    # Phase 2 answers moves only; drop anything that is not one.
                    ids = [i for i in ids if low <= i < high]
        steps = tokenizer.decode(ids, chunk)
        if steps is None:
            return []
        return [np.array([dx * 100, dy * 100, dz * 100, dyaw]) for dx, dy, dz, dyaw in steps]

    return predict


@app.route("/reset", methods=["POST"])
def reset():
    # The evaluator calls /reset before every task: a new rollout episode.
    STATE["episode"] += 1
    STATE["call"] = 0
    STATE["goal"], STATE["yaw_total"], STATE["last_yaw"] = None, 0.0, None
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    image = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")
    proprio = np.array(data["proprio"], dtype=float)  # x, y, z (sim units), yaw (deg)
    if not proprio.any():  # first call of a task: keep what the drone saw at the start
        STATE["first_frames"] += 1
        image.save(STATE["frames_dir"] / f"{STATE['first_frames']:03d}.png")
    # Summed yaw since the start (D175 goal memory needs turns beyond 180 degrees).
    if STATE["last_yaw"] is not None:
        STATE["yaw_total"] += (proprio[3] - STATE["last_yaw"] + 180.0) % 360.0 - 180.0
    STATE["last_yaw"] = float(proprio[3])
    steps = STATE["predict"](image, proprio, data["instr"])
    stop = bool(steps) and all(
        np.linalg.norm(s[:3]) < STILL_POS and abs(math.degrees(s[3])) < STILL_YAW_DEG for s in steps
    )
    if STATE["stop_confirm"] and stop:
        steps = steps[:1]  # one still move, then ask again (D168)
    if STATE["execute_steps"]:
        steps = steps[: STATE["execute_steps"]]  # re-ask with a new photo sooner (D173)
    poses, position, yaw = [], proprio[:3].copy(), math.radians(proprio[3])
    for step in steps:
        c, s = math.cos(yaw), math.sin(yaw)
        position = position + np.array(
            [c * step[0] - s * step[1], s * step[0] + c * step[1], step[2]]
        )
        yaw = yaw + float(step[3])
        poses.append([*position.tolist(), yaw])
    record = dict(instr=data["instr"], proprio=proprio.tolist(), poses=poses, stop=stop)
    record |= STATE.pop("progress", {})
    if "box" in STATE:
        record["box"] = STATE.pop("box")
    if STATE["rollout_dir"]:
        # D172: everything GRPO needs to re-score this call under a newer policy.
        name = f"ep{STATE['episode']:05d}_c{STATE['call']:03d}.png"
        image.save(STATE["rollout_dir"] / name)
        record |= dict(episode=STATE["episode"], call=STATE["call"], image=name, **STATE.pop("sample", {}))
        STATE["call"] += 1
    STATE["log"].write(json.dumps(record) + "\n")
    STATE["log"].flush()
    return jsonify({"status": "success", "action": poses})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["openvla-uav", "qwen"])
    parser.add_argument("--path", default="D:/drone_vla_pilot/models/openvla-uav")
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--precision", default="nf4", choices=["nf4", "bf16"])
    parser.add_argument("--port", type=int, default=5007)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--stop-confirm",
        action="store_true",
        help="when every step of a chunk is still, return only the first one",
    )
    parser.add_argument(
        "--rollout-temperature",
        type=float,
        help="qwen only (D172 RL): sample action tokens at this temperature and save "
        "each call's image, prompt, token ids and log-probabilities",
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        help="D173: fly only the first N of the K predicted moves, then ask again",
    )
    parser.add_argument(
        "--contrast-alpha",
        type=float,
        help="D173: contrastive decoding against a gray photo with this strength",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="D175: the adapter writes a progress line before its moves (train --progress)",
    )
    parser.add_argument(
        "--goal-memory",
        action="store_true",
        help="D175: keep the first progress line as a goal and hand it back every call",
    )
    parser.add_argument(
        "--save-call-images",
        action="store_true",
        help="D185 DAgger: save every call's photo and episode/call numbers also when decoding "
        "greedily (without --rollout-temperature)",
    )
    parser.add_argument(
        "--memory-deadband",
        help="D182 with --goal-memory: 'metres,degrees'; inside it the line becomes 0 (arrived)",
    )
    parser.add_argument(
        "--box",
        action="store_true",
        help="D193: box-first adapter - the model writes the target's bbox_2d before its progress line",
    )
    parser.add_argument(
        "--unconstrained-greedy",
        action="store_true",
        help="D191: the old greedy decoding (any token, non-action tokens dropped afterwards); "
        "only to reproduce runs before D191",
    )
    args = parser.parse_args()
    assert args.progress or not args.goal_memory, "--goal-memory needs --progress"
    deadband = [float(v) for v in args.memory_deadband.split(",")] if args.memory_deadband else None
    STATE["stop_confirm"] = args.stop_confirm
    STATE["goal"], STATE["yaw_total"], STATE["last_yaw"] = None, 0.0, None
    STATE["execute_steps"] = args.execute_steps
    STATE["episode"], STATE["call"] = 0, 0
    STATE["rollout_dir"] = None
    if args.rollout_temperature or args.save_call_images:
        STATE["rollout_dir"] = args.log.parent / "rollout_images"
        STATE["rollout_dir"].mkdir(parents=True, exist_ok=True)
    if args.model == "openvla-uav":
        STATE["predict"] = openvla_loader(args.path, args.precision)
    else:
        STATE["predict"] = qwen_loader(
            args.path, args.chunk, args.precision, args.rollout_temperature, args.contrast_alpha,
            progress=args.progress, goal_memory=args.goal_memory, deadband=deadband,
            constrain=not args.unconstrained_greedy, box=args.box,
        )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    STATE["log"] = open(args.log, "a", encoding="utf-8")  # noqa: SIM115 - lives with the server
    STATE["frames_dir"] = args.log.parent / "first_frames"
    STATE["frames_dir"].mkdir(exist_ok=True)
    STATE["first_frames"] = 0
    app.run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
