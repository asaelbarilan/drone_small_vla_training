"""D193: box labels for "box first" training - the BASE Qwen3-VL-4B (no adapter) boxes the
object each instruction refers to.

D192 found that our fine-tuned model writes a flight answer whatever it is asked: the base
model's grounding output was overwritten by the action training. The base model still boxes
people 8/8 on our validation photos (and found 679 of 733 targets in D189), so it labels the
training photos; the VLA is then trained to write that box before its progress line (ECoT,
2407.08693; ECoT-Lite 2505.08243: the box can also be left out at test time).

Which frames: simulator flights (the test domain): frame 0, every --every-th frame and the
last one; real flights: frame 0, the middle and the last frame (the first call is the one that
fixes the goal under goal memory, D182). --max-seconds stops labelling in time; frames left
without a label are simply trained without a box (ECoT-Lite's reasoning dropout). The model is asked
for the object the INSTRUCTION refers to, so any wording works ("the middle tree", "the
streetlight"); instructions without an object (rotate 90 degrees, ascend 5 m) and objects that
are not visible get an empty box. Coordinates are Qwen's native 0-1000 frame of the image.

Output: --out .jsonl, one line per frame {"image": <store path as in episodes.jsonl>,
"box": [x1, y1, x2, y2] or []}; resumable (frames already in --out are skipped).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_uav_flow_vla import OFFICIAL_STORE, image_path  # noqa: E402

PROMPT = (
    'A drone was told: "{instruction}". Locate the object this instruction refers to in the '
    "image (dogs in this world can be four-legged robot dogs). Output its bounding box as JSON: "
    '{{"bbox_2d": [x1, y1, x2, y2]}}. If the instruction names no object, or the object is not '
    'visible, output {{"bbox_2d": []}}.'
)
# Instructions that only give a number (rotate / shift / ascend by an amount) name no object.
NO_OBJECT = re.compile(
    r"^(please |need to |request to |suggest )?(rotate|turn (left|right|around)|ascend|descend|"
    r"climb|rise|lower|move \d|fly \d|go up|go down|shift)\b",
    re.IGNORECASE,
)


def parse_box(text):
    m = re.search(r'"bbox_2d"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]', text)
    if not m:
        return []
    x1, y1, x2, y2 = (min(max(int(v), 0), 1000) for v in m.groups())
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="base Qwen3-VL-4B-Instruct folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--every", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="at most this many frames (0 = all)")
    parser.add_argument("--max-seconds", type=int, default=0, help="stop after this long (0 = no limit)")
    args = parser.parse_args()
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    done = set()
    if args.out.exists():
        done = {json.loads(line)["image"] for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = []
    for line in (OFFICIAL_STORE / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        episode = json.loads(line)
        n = len(episode["images"])
        sim = episode["episode"].startswith("sim_")
        frames = {0, n - 1, *range(0, n, args.every)} if sim else {0, n // 2, n - 1}
        for t in sorted(frames):
            image = episode["images"][t]
            if image not in done:
                todo.append((image, episode["instruction"]))
    # Simulator frames first: they are the test domain, and --max-seconds may stop early.
    todo.sort(key=lambda item: "sim_" not in item[0].replace("\\", "/"))
    if args.limit:
        todo = todo[: args.limit]
    print(json.dumps(dict(frames_to_label=len(todo), already=len(done))), flush=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    start = time.monotonic()
    with args.out.open("a", encoding="utf-8") as out:
        for i in range(0, len(todo), args.batch):
            if args.max_seconds and time.monotonic() - start > args.max_seconds:
                print(json.dumps(dict(stopped_at=i, reason="max-seconds")), flush=True)
                break
            part = todo[i : i + args.batch]
            boxes = {}
            ask = []
            for image, instruction in part:
                if NO_OBJECT.search(instruction.strip()):
                    boxes[image] = []
                else:
                    ask.append((image, instruction))
            if ask:
                pictures = [Image.open(image_path(dict(image=im, format="official"))).convert("RGB") for im, _ in ask]
                texts = [
                    processor.apply_chat_template(
                        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(instruction=ins)}]}],
                        tokenize=False, add_generation_prompt=True,
                    )
                    for _, ins in ask
                ]
                batch = processor(text=texts, images=pictures, return_tensors="pt", padding=True).to("cuda")
                with torch.inference_mode():
                    ids = model.generate(**batch, max_new_tokens=40, do_sample=False)
                answers = processor.batch_decode(ids[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
                for (image, _), answer in zip(ask, answers, strict=True):
                    boxes[image] = parse_box(answer)
            for image, _ in part:
                out.write(json.dumps(dict(image=image, box=boxes[image])) + "\n")
            out.flush()
            if (i // args.batch) % 50 == 0:
                print(json.dumps(dict(labelled=i + len(part), of=len(todo))), flush=True)


if __name__ == "__main__":
    main()
