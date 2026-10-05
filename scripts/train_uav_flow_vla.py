"""D155: UAV-Flow chunk training with the official action-token representation.

Follows OpenVLA-UAV/vla-scripts/finetune_uav.py where the hardware allows: 256-bin
action tokens on the vocabulary tail, LoRA rank 32, learning rate 5e-4, no image
augmentation. It cannot follow the base model - the recipe fine-tunes openvla-7b
across 8 GPUs, and this runs a 256M/500M VLM on one 8 GB card. That substitution
is deliberate and no number from here is comparable to a published one.

Action tokens are spliced into `input_ids` by id. They are NOT rendered to text
and re-encoded: a measured 177 of 200 chunks come back as different ids that way,
because the repurposed vocabulary tail holds ordinary word-pieces that BPE
re-merges. The official implementation works on ids for the same reason.
"""

import argparse
import contextlib
import importlib
import itertools
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import (
    LoraConfig,
    get_peft_model,
    load_peft_weights,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from PIL import Image, ImageOps
from uav_flow_action_tokenizer import ActionTokenizer, load_stats

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/uav_flow_chunks_20260921"
STORE = Path(os.environ.get("UAV_FLOW_STORE", "D:/drone_vla_pilot/data/uav_flow_chunks_20260921"))
PROMPT = "Drone forward camera. What action should the drone take to {instruction}?"
# D158: the official OpenVLA-UAV format (prepare_uav_flow_official.py).
# D175: the 10-shard manifest (action range 0.1/99.9%) is the one every adapter since
# D160 was trained and decoded with; machines used to sed-edit this line by hand.
# The 1-shard manifest (reports/uav_flow_official_20260922) is still selectable.
OFFICIAL_REPORT = Path(
    os.environ.get("UAV_FLOW_OFFICIAL_REPORT", ROOT / "reports/uav_flow_official_10shard")
)
OFFICIAL_STORE = Path(
    os.environ.get("UAV_FLOW_OFFICIAL_STORE", "D:/drone_vla_pilot/data/uav_flow_official_20260922")
)
OFFICIAL_PROMPT = "Current State: {state}, What action should the uav take to {instruction}?"
OFFICIAL_REPEAT = 5
# D175 progress tokens: before its moves the model writes what is left to the end of
# the flight, in its own frame, as fixed-width text (same token count every time):
# "Left +12.3,-00.4,+00.0,-090" = metres forward, sideways, up, then degrees of yaw.
PROGRESS_PATTERN = re.compile(
    r"Left ([+-]\d\d\.\d),([+-]\d\d\.\d),([+-]\d\d\.\d),([+-]\d\d\d)"
)
# D193 box-first line, Qwen's native grounding JSON: {"bbox_2d": [x1, y1, x2, y2]} or [].
BOX_PATTERN = re.compile(r'"bbox_2d"\s*:\s*\[([^\]]*)\]')
SMOL500 = "D:/drone_vla_pilot/models/SmolVLM-500M-Instruct/a7da5b986cb59b408707209984f360a5f4ad7e47"


def setup(model_name, precision="nf4"):
    """Load a base model the way this repository already loads it, and say which
    layer prefix LoRA should attach to. Qwen is 4-bit with gradient checkpointing,
    which leaves its vision tower frozen - the expert-only regime Exp2VLA uses.

    precision="bf16" loads the unquantised base (cloud GPUs). D157 found about
    40% of the PyTorch-vs-llama.cpp token disagreement came from quantisation,
    and the NF4 reference could not separate the rest; training against the true
    weights removes NF4 from the chain."""
    api = importlib.import_module(
        # qwen_backend = the same loading code without the architecture testbed (D186).
        "qwen_backend" if model_name == "qwen" else "run_smol_frd_overfit"
    )
    if model_name == "smol500":
        api.MODEL = Path(SMOL500)
    if model_name == "qwen" and precision == "bf16":
        processor = api.AutoProcessor.from_pretrained(str(api.MODEL), local_files_only=True)
        model = api.Qwen3VLForConditionalGeneration.from_pretrained(
            str(api.MODEL),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()},
            attn_implementation="sdpa",
        )
        model.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        return api, processor, model, "language_model.layers."
    if model_name == "qwen":
        processor = api.AutoProcessor.from_pretrained(str(api.MODEL), local_files_only=True)
        model = prepare_model_for_kbit_training(
            api.load_base(),
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        # run_qwen_frd_overfit.load_base upcasts norm weights to fp32, which the
        # bf16 activations here cannot feed. Keep the whole frozen base in bf16.
        for param in model.parameters():
            if param.dtype == torch.float32 and not param.requires_grad:
                param.data = param.data.to(torch.bfloat16)
        return api, processor, model, "language_model.layers."
    processor = api.load_processor()
    model = api.load_base()
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return api, processor, model, "text_model.layers."


SIDES = {
    "left": "right",
    "right": "left",
    "leftward": "rightward",
    "rightward": "leftward",
    "clockwise": "counterclockwise",
    "counterclockwise": "clockwise",
    "anticlockwise": "clockwise",
}
# longest first, so "counterclockwise" is matched before "clockwise"
SIDE_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(SIDES, key=len, reverse=True)) + r")\b", re.IGNORECASE
)


def remaining_to_end(proprio, t):
    """D175: displacement from frame t to the flight's last frame, in the drone's
    frame at t (the rotation the actions use), yaw in degrees. `proprio` is the
    per-frame state the prompt shows: x, y, z in metres, yaw in degrees. The yaw
    left is the summed per-frame turn, so a full circle reads 360, not 0."""
    here, end = proprio[t], proprio[-1]
    yaw = math.radians(here[3])
    dx, dy = end[0] - here[0], end[1] - here[1]
    forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    sideways = -math.sin(yaw) * dx + math.cos(yaw) * dy
    turn = sum(
        (after[3] - before[3] + 180.0) % 360.0 - 180.0
        for before, after in zip(proprio[t:-1], proprio[t + 1 :])
    )
    return [forward, sideways, end[2] - here[2], turn]


def progress_text(left):
    """Fixed-width progress line; metres clipped to +-99.9, yaw to +-360."""
    metres = [min(max(v, -99.9), 99.9) for v in left[:3]]
    turn = int(round(min(max(left[3], -360.0), 360.0)))
    return "Left " + ",".join(f"{v:+05.1f}" for v in metres) + f",{turn:+04d}\n"


def parse_progress(text):
    """Progress line -> [forward, sideways, up, yaw_deg], or None if malformed."""
    match = PROGRESS_PATTERN.search(text or "")
    return [float(g) for g in match.groups()] if match else None


def box_text(box):
    """D193 box-first: the target's box in Qwen's native JSON (0-1000 image frame), written
    before the progress line; an empty list when the instruction names no visible object."""
    return '{"bbox_2d": [' + ", ".join(str(int(v)) for v in box) + "]}\n"


def parse_box(text):
    """The first bbox_2d in text -> [x1, y1, x2, y2] (0-1000), [] for an empty box, None if absent."""
    match = BOX_PATTERN.search(text or "")
    if not match:
        return None
    values = [int(v) for v in re.findall(r"-?\d+", match.group(1))]
    return values if len(values) == 4 else []


def mirror_box(box):
    """Left-right flip in the 0-1000 frame."""
    if not box:
        return box
    x1, y1, x2, y2 = box
    return [1000 - x2, y1, 1000 - x1, y2]


def mirror_instruction(text):
    """Swap the side words in one pass, keeping capitalisation."""

    def swap(match):
        word = match.group(0)
        other = SIDES[word.lower()]
        return other.capitalize() if word[0].isupper() else other

    return SIDE_PATTERN.sub(swap, text)


def mirror_row(row):
    """A left-right mirrored copy: photo flipped at load, sideways and yaw negated,
    side words swapped. Free data that teaches exactly the grounding we fail."""
    return {
        **row,
        "id": row["id"] + ":m",
        "mirror": True,
        "instruction": mirror_instruction(row["instruction"]),
        "prompt": mirror_instruction(row["prompt"]),
        "chunk": [[dx, -dy, dz, -dyaw] for dx, dy, dz, dyaw in row["chunk"]],
        **mirror_progress(row),
        **({"box": mirror_box(row["box"])} if "box" in row else {}),
    }


def mirror_progress(row):
    if "progress" not in row:
        return {}
    forward, sideways, up, turn = row["progress"]
    return {"progress": [forward, -sideways, up, -turn]}


def image_path(row):
    """Frames are stored with absolute Windows paths; on another machine the same
    frame lives under UAV_FLOW_STORE/frames/<episode>/<file>."""
    path = Path(row["image"].replace("\\", "/"))
    if path.exists():
        return path
    store = OFFICIAL_STORE if row.get("format") == "official" else STORE
    return store / "frames" / path.parent.name / path.name


def motion_kind(text):
    """Coarse motion type of an instruction, for per-type validation (D169)."""
    text = text.lower()
    for kind, pattern in (
        ("Land", r"\bland"),
        ("Pass", r"through|\bpass\b|proceed"),
        ("Surround", r"circle|orbit|around|surround"),
        ("Ascend/Descend", r"ascend|descend|climb|altitude|rise|lower"),
        ("Retreat", r"back from|away from|retreat|move back"),
        ("Approach/Move", r"approach|close to|navigate to a point|fly to|move to|go to"),
        ("Turn/Rotate", r"turn|rotate|face|clockwise"),
        ("Shift", r"meters? (to the )?(left|right|forward)|degree"),
    ):
        if re.search(pattern, text):
            return kind
    return "other"


def hold_out_sim_flights(rows, per_kind, seed=17):
    """D169: the simulator set ships no split, so hold out `per_kind` simulator
    flights of every motion type for validation in the test domain."""
    kinds = {}
    for row in rows:
        if row["episode"].startswith("sim_") and not row.get("mirror"):
            kinds.setdefault(row["episode"], motion_kind(row["instruction"]))
    by_kind = {}
    for episode, kind in sorted(kinds.items()):
        by_kind.setdefault(kind, []).append(episode)
    picker = random.Random(seed)
    return {e for group in by_kind.values() for e in picker.sample(group, min(per_kind, len(group)))}


def official_rows(
    split, k, instruction="instruction", oversample=False, mirror=False, progress=False
):
    """Per-frame examples in the official format, with a K-step target.

    The target is actions[t : t+K], each step in the drone frame at that step; past
    the end of the flight it is padded with zero actions, which is what the
    official data uses for the last frame. K=1 is exactly the official sample.
    `instruction`: "instruction" (official, free wording), "instruction_unified",
    or "both" (alternates by frame, so each flight is seen with both wordings).
    With oversample=True the first and last frames are repeated 5 extra times.
    progress=True (D175) adds the displacement left to the flight's end as a target.
    """
    rows = []
    for line in (OFFICIAL_STORE / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        episode = json.loads(line)
        if episode["split_unseen"] != split:
            continue
        actions, n = episode["actions"], len(episode["actions"])
        for t in range(n):
            if instruction == "both":
                key = "instruction" if t % 2 == 0 else "instruction_unified"
            else:
                key = instruction
            chunk = actions[t : t + k]
            chunk = chunk + [[0.0] * 4] * (k - len(chunk))
            state = ",".join(str(round(float(x), 1)) for x in episode["proprio"][t])
            row = dict(
                format="official",
                id=f"{episode['episode']}:{t:05d}",
                episode=episode["episode"],
                step=t,
                image=episode["images"][t],
                prompt=OFFICIAL_PROMPT.format(state=state, instruction=episode[key]),
                instruction=episode[key],
                chunk=chunk,
                chunk_len=k,
            )
            if progress:
                row["progress"] = remaining_to_end(episode["proprio"], t)
            repeats = 1 + (OFFICIAL_REPEAT if oversample and t in (0, n - 1) else 0)
            rows.extend([row] * repeats)
            if mirror:
                rows.extend([mirror_row(row)] * repeats)
    return rows


def official_stats():
    manifest = json.loads((OFFICIAL_REPORT / "manifest.json").read_text(encoding="utf-8"))
    return manifest["action_stats_train"]


def load_image(row, gray=False):
    picture = Image.open(image_path(row)).convert("RGB")
    picture.thumbnail((256, 256))
    if row.get("mirror"):
        picture = ImageOps.mirror(picture)
    return Image.new("RGB", picture.size, (127, 127, 127)) if gray else picture


def encode(processor, tokenizer, row, k, gray=False, with_answer=True, image=None):
    image = load_image(row, gray) if image is None else image
    text = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": row["prompt"]
                        if "prompt" in row
                        else PROMPT.format(instruction=row["instruction"].rstrip(".").lower()),
                    },
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    batch = processor(text=[text], images=[image], return_tensors="pt")
    box = box_ids(processor, row)
    progress = progress_ids(processor, row)
    if not with_answer:
        # Generation continues after the given box (D193) and progress line (server phase 2, D175).
        batch = append_ids(batch, box)
        return append_ids(batch, progress) if "progress_text" in row else batch
    prompt_length = batch["input_ids"].shape[1]
    if row.get("progress_context"):
        # Rollouts (D175): the progress line was the server's, the actions are the
        # policy's; the line is context, only the actions are scored.
        batch = append_ids(batch, progress)
        prompt_length, progress = batch["input_ids"].shape[1], []
    ids = [*box, *progress, *tokenizer.token_ids(row["chunk"][:k]), processor.tokenizer.eos_token_id]
    batch = append_ids(batch, ids)
    labels = batch["input_ids"].clone()
    labels[:, :prompt_length] = -100
    batch["labels"] = labels
    return batch


def box_ids(processor, row):
    """D193: the box line as text ids (`box_text` from the server, or the `box` label); []
    when the row has no box (unlabelled frames train without one, ECoT-Lite's dropout)."""
    if "box_text" in row:
        text = row["box_text"]
    elif "box" in row:
        text = box_text(row["box"])
    else:
        return []
    return processor.tokenizer.encode(text, add_special_tokens=False)


def progress_ids(processor, row):
    """D175: the progress line as text ids, from text the server generated
    (`progress_text`) or from the label (`progress`); [] when there is none."""
    if "progress_text" in row:
        text = row["progress_text"]
    elif "progress" in row:
        text = progress_text(row["progress"])
    else:
        return []
    return processor.tokenizer.encode(text, add_special_tokens=False)


def append_ids(batch, ids):
    """Extend one encoded example by plain-text token ids."""
    if not ids:
        return batch
    answer = torch.tensor([ids], dtype=batch["input_ids"].dtype)
    batch["input_ids"] = torch.cat([batch["input_ids"], answer], dim=1)
    batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(answer)], dim=1)
    # transformers 5 also returns per-token type ids (mm_token_type_ids); the
    # answer is plain text, which is type 0.
    for key in ("mm_token_type_ids", "token_type_ids"):
        if key in batch:
            batch[key] = torch.cat([batch[key], torch.zeros_like(answer)], dim=1)
    return batch


# Per-token fields and their padding value (None = the tokenizer's pad id).
# Everything else in an encoded example (pixel_values, image_grid_thw, ...) is
# per-image and concatenates as is.
SEQUENCE_FILL = dict(
    input_ids=None, attention_mask=0, labels=-100, mm_token_type_ids=0, token_type_ids=0
)


def collate_left(items, pad_id):
    """Left-padded batch for generation: decoder-only models continue from the
    last position, so the padding has to sit in front, not behind."""
    width = max(item["input_ids"].shape[1] for item in items)
    batch = {}
    for key in items[0]:
        if key in SEQUENCE_FILL and key != "labels":
            fill = pad_id if SEQUENCE_FILL[key] is None else SEQUENCE_FILL[key]
            parts = [F.pad(i[key], (width - i[key].shape[1], 0), value=fill) for i in items]
        elif key == "labels":
            continue
        else:
            parts = [i[key] for i in items]
        batch[key] = torch.cat(parts, dim=0)
    return batch


def collate(items, pad_id):
    """Right-pad single-example encodings into one batch. Every answer is the same
    49 tokens, so the model's token-mean loss is also the per-example mean."""
    width = max(item["input_ids"].shape[1] for item in items)
    batch = {}
    for key in items[0]:
        if key in SEQUENCE_FILL:
            fill = pad_id if SEQUENCE_FILL[key] is None else SEQUENCE_FILL[key]
            parts = [F.pad(i[key], (0, width - i[key].shape[1]), value=fill) for i in items]
        else:
            parts = [i[key] for i in items]
        batch[key] = torch.cat(parts, dim=0)
    return batch


class Examples(torch.utils.data.Dataset):
    """Encodes on DataLoader workers so image decoding overlaps the GPU step."""

    def __init__(self, processor, tokenizer, rows, k, photo_aug=False):
        self.processor, self.tokenizer, self.rows, self.k = processor, tokenizer, rows, k
        self.photo_aug = photo_aug

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if not self.photo_aug:
            return encode(self.processor, self.tokenizer, row, self.k)
        # D193: the evaluator hands the model the simulator's BGR array as RGB, at 224 px,
        # while the stored photos are RGB at 256 px (D192). Train on both, so the model sees
        # what it gets in flight and keeps what it learned on the stored photos.
        image = load_image(row)
        if random.random() < 0.5:
            image = Image.fromarray(np.asarray(image)[:, :, ::-1].copy())
        if random.random() < 0.5:
            image = image.resize((224, 224))
        return encode(self.processor, self.tokenizer, row, self.k, image=image)


class Collate:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, items):
        return collate(items, self.pad_id)


class ExampleOrder(torch.utils.data.Sampler):
    """The same shuffled-epoch stream the batch-1 trainer used (a fresh
    permutation per epoch, consumed from the end), resumable by skipping the
    examples already seen."""

    def __init__(self, n, seed, skip, rank=0, world=1):
        self.n, self.seed, self.skip, self.rank, self.world = n, seed, skip, rank, world

    def __iter__(self):
        rng = random.Random(self.seed)

        def stream():
            while True:
                yield from reversed(rng.sample(range(self.n), self.n))

        # Under torchrun every rank walks the same global stream and takes every
        # world-th example, so the ranks never see the same example in one update.
        return itertools.islice(stream(), self.skip + self.rank, None, self.world)


def distributed():
    """(rank, local rank, world size); a no-op unless launched with torchrun."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if world > 1:
        torch.distributed.init_process_group("nccl")
    return int(os.environ.get("RANK", "0")), local, world


def learning_rate(args, step, report=None):
    if args.schedule == "constant":
        return args.lr
    warmup = max(1, round(args.warmup * args.updates))
    if args.schedule == "wsd":
        # D176 warmup-stable-decay: hold args.lr until real validation accuracy goes
        # flat (or the cap gets close), then a cosine decay over --decay-updates.
        decay_from = (report or {}).get("decay_from")
        if decay_from is None:
            return args.lr * min(1.0, step / warmup) if args.warmup else args.lr
        progress = min(1.0, (step - decay_from) / max(1, args.decay_updates))
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))
    if step <= warmup:
        return args.lr * step / warmup
    progress = (step - warmup) / max(1, args.updates - warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen", choices=["smol256", "smol500", "qwen"])
    parser.add_argument("--split", default="split_unseen", choices=["split_unseen", "split_seen"])
    parser.add_argument("--chunk", type=int, default=8, help="prediction horizon K (steps)")
    parser.add_argument(
        "--format",
        default="d155",
        choices=["d155", "official"],
        help="official = OpenVLA-UAV: 4-D drone-frame actions, state in the prompt",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="official format: add a left-right mirrored copy of every training example",
    )
    parser.add_argument(
        "--instruction",
        default="instruction",
        choices=["instruction", "instruction_unified", "both"],
        help="official format only; the official code uses `instruction`",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="D175 official format: write the displacement left to the flight's end "
        "(progress tokens) before the moves",
    )
    parser.add_argument(
        "--extra-rows",
        nargs="*",
        type=Path,
        default=[],
        help="D185 DAgger: .jsonl training rows (dagger_relabel.py) added to the training set",
    )
    parser.add_argument("--extra-repeat", type=int, default=1, help="copies of each extra row")
    parser.add_argument(
        "--boxes",
        type=Path,
        help="D193 box-first: label_boxes.py output; labelled frames write the target's box "
        "before the progress line",
    )
    parser.add_argument(
        "--box-repeat",
        type=int,
        default=1,
        help="D193: training copies of each box-labelled row (the labelled frames are a small share)",
    )
    parser.add_argument(
        "--photo-aug",
        action="store_true",
        help="D193: random BGR swap and 224 px resize of training photos (what the evaluator sends)",
    )
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1, help="examples per forward pass")
    parser.add_argument("--accum", type=int, default=8, help="forward passes per update")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader encode workers")
    parser.add_argument("--schedule", default="constant", choices=["constant", "cosine", "wsd"])
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=3,
        help="wsd: start the decay after this many validations in a row with no new best "
        "real-flight token accuracy (exact or within one bin)",
    )
    parser.add_argument("--decay-updates", type=int, default=3000, help="wsd: length of the decay")
    parser.add_argument(
        "--plateau-sim",
        action="store_true",
        help="wsd: a new best simulator accuracy also counts as progress (D177)",
    )
    parser.add_argument(
        "--decay-now",
        action="store_true",
        help="wsd on --resume: start the final decay at the checkpoint step (D179)",
    )
    parser.add_argument(
        "--reset-plateau",
        action="store_true",
        help="wsd on --resume: judge the plateau only on validations after this resume "
        "(D176: progress tokens switched on mid-run may dip move accuracy at first)",
    )
    parser.add_argument("--warmup", type=float, default=0.03, help="cosine warmup fraction")
    parser.add_argument("--save-every", type=int, default=0, help="checkpoint every N updates")
    parser.add_argument("--resume", action="store_true", help="continue from OUT/checkpoint")
    parser.add_argument(
        "--init-adapter", type=Path, help="initialise LoRA weights from this adapter"
    )
    parser.add_argument("--epochs", type=float, help="set --updates from passes over train")
    parser.add_argument("--wandb-project", help="log to Weights & Biases (needs WANDB_API_KEY)")
    parser.add_argument("--wandb-entity", default="asael", help="team that owns the project")
    parser.add_argument("--lr", type=float, default=5e-4, help="official recipe value")
    parser.add_argument("--lora-rank", type=int, default=32, help="official recipe value")
    parser.add_argument(
        "--targets",
        default="all-linear",
        choices=["all-linear", "text-only"],
        help="all-linear matches the recipe and trains the visual path too",
    )
    parser.add_argument(
        "--precision",
        default="nf4",
        choices=["nf4", "bf16"],
        help="bf16 = unquantised base, needs about 9 GB more VRAM than nf4",
    )
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--val-batches", type=int, default=160)
    parser.add_argument(
        "--sim-val-per-kind",
        type=int,
        default=0,
        help="official format: hold out this many simulator flights per motion type",
    )
    parser.add_argument("--sim-val-examples", type=int, default=1024)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="stop after this many validations in a row without a new best (0 = off)",
    )
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rank, local, world = distributed()
    main_rank = rank == 0
    if main_rank:
        args.out.mkdir(parents=True, exist_ok=args.resume)
    checkpoint = args.out / "checkpoint"
    start = time.monotonic()
    sim_held = []

    if args.format == "official":
        assert args.split == "split_unseen", "the official data carries the unseen split only"
        manifest = json.loads((OFFICIAL_REPORT / "manifest.json").read_text(encoding="utf-8"))
        manifest["task"] = (
            f"current frame + state + instruction -> next {args.chunk} (dx,dy,dz,dyaw), drone frame"
        )
        manifest["steps_sha256"] = manifest["episodes_sha256"]
        baselines = None
        stats = official_stats()
        train = official_rows(
            "train",
            args.chunk,
            args.instruction,
            oversample=True,
            mirror=args.mirror,
            progress=args.progress,
        )
        held = official_rows("val", args.chunk, args.instruction, progress=args.progress)
        if args.sim_val_per_kind:
            held_ids = hold_out_sim_flights(train, args.sim_val_per_kind)
            train = [r for r in train if r["episode"] not in held_ids]
            sim_held = [
                r
                for r in official_rows(
                    "train", args.chunk, args.instruction, progress=args.progress
                )
                if r["episode"] in held_ids
            ]
            if main_rank:
                (args.out / "sim_val_flights.json").write_text(
                    json.dumps(sorted(held_ids)), encoding="utf-8"
                )
    else:
        manifest = json.loads((REPORT / "manifest.json").read_text(encoding="utf-8"))
        baselines = json.loads((REPORT / "baselines.json").read_text(encoding="utf-8"))[args.split]
        stats = load_stats(args.split)
        rows = [
            json.loads(s) for s in (STORE / "steps.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        train = [r for r in rows if r[args.split] == "train" and r["chunk_len"] == args.chunk]
        held = [r for r in rows if r[args.split] == "val" and r["chunk_len"] == args.chunk]
    if args.boxes:
        labels = {}
        for line in args.boxes.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                labels[item["image"]] = item["box"]
        boxed = 0
        for r in [*train, *held, *sim_held]:
            if r["image"] in labels:
                box = labels[r["image"]]
                r["box"] = mirror_box(box) if r.get("mirror") else box
                boxed += 1
        extra_boxed = [r for r in train if "box" in r] * (args.box_repeat - 1)
        train += extra_boxed
        if main_rank:
            print(json.dumps(dict(box_labels=len(labels), rows_with_box=boxed, box_repeats_added=len(extra_boxed))), flush=True)
    for path in args.extra_rows:
        extra = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        train += extra * args.extra_repeat
        if main_rank:
            print(json.dumps(dict(extra_rows=str(path), rows=len(extra), repeat=args.extra_repeat)), flush=True)
    if not held:
        # A store without validation flights (the Windows RL anchor store): validate on a
        # fixed sample of training rows, so the curves still show the fit (D185).
        held = random.Random(5).sample(train, min(64, len(train)))
    assert train and held, "need full-length chunks on both sides"
    # A fixed held-out sample, so every point on the curve is the same examples.
    held = random.Random(9).sample(held, min(args.val_batches, len(held)))
    sim_held = random.Random(9).sample(sim_held, min(args.sim_val_examples, len(sim_held)))
    per_update = args.batch_size * args.accum * world
    if args.epochs:
        args.updates = math.ceil(args.epochs * len(train) / per_update)

    api, processor, model, prefix = setup(args.model, args.precision)
    globals()["api"] = api
    tokenizer = ActionTokenizer(processor.tokenizer, stats)
    # The official recipe uses target_modules="all-linear", which covers the vision
    # encoder and the projector. Restricting LoRA to the language layers leaves the
    # whole visual path frozen, and the model then learns a text-to-action lookup:
    # the D155 "b" run scored 2.86 m on episodes it trained on, 12.09 m held out,
    # and moved 1 cm when its images were replaced by flat gray.
    if args.targets == "all-linear":
        targets = "all-linear"
        covered = sorted(
            {
                ".".join(n.split(".")[:3])
                for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear)
            }
        )
    else:
        targets = [
            n
            for n, _ in model.named_modules()
            if prefix in n
            and n.rsplit(".", 1)[-1]
            in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        ]
        assert targets
        covered = ["text_model"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=32,
            lora_dropout=0,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    if args.init_adapter:
        # D165: start from an earlier adapter (fresh optimiser and schedule), e.g.
        # to continue the real-data model on real + simulator flights.
        set_peft_model_state_dict(model, load_peft_weights(str(args.init_adapter)))
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=args.lr)

    report = dict(
        status="running",
        model=args.model,
        base=str(api.MODEL),
        recipe=dict(
            follows="OpenVLA-UAV finetune_uav.sh",
            action_tokens=256,
            lora_rank=args.lora_rank,
            learning_rate=args.lr,
            image_aug=False,
            target_modules=args.targets,
            deviation="base model is SmolVLM-256M on one GPU, not openvla-7b on eight",
        ),
        split=args.split,
        chunk=args.chunk,
        task=manifest["task"],
        baselines=baselines,
        steps_sha256=manifest["steps_sha256"],
        action_stats=stats,
        data_format=args.format,
        init_adapter=str(args.init_adapter) if args.init_adapter else None,
        mirrored=args.mirror,
        boxes=str(args.boxes) if args.boxes else None,
        photo_aug=args.photo_aug,
        instruction_field=args.instruction if args.format == "official" else "instruction_unified",
        trainable_module_roots=covered,
        train_examples=len(train),
        held_out_examples=len(held),
        sim_val_per_kind=args.sim_val_per_kind,
        sim_val_examples=len(sim_held),
        updates=args.updates,
        precision=args.precision,
        accum=args.accum,
        batch_size=args.batch_size,
        world_size=world,
        examples_per_update=per_update,
        epochs=round(args.updates * per_update / len(train), 3),
        schedule=args.schedule,
        losses=[],
        seconds_per_update=[],
        held_out=[],
    )

    def save():
        if not main_rank:
            return
        report["elapsed_s"] = round(time.monotonic() - start, 1)
        (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    first = 1
    if args.resume:
        state = torch.load(checkpoint / "state.pt", weights_only=False)
        set_peft_model_state_dict(model, load_peft_weights(str(checkpoint)))
        optimiser.load_state_dict(state["optimiser"])
        report = state["report"]
        first = state["step"] + 1
        report.setdefault("resumed_at", []).append(state["step"])
        if args.reset_plateau:
            report["real_best"] = dict(token_acc=-1.0, token_acc_1bin=-1.0)
            report["real_flat"] = 0
        if args.decay_now:
            report["decay_from"], report["decay_reason"] = state["step"], "user: decay now"

    core = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])

    run = None
    if args.wandb_project and main_rank:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.out.name,
            id=args.out.name,
            resume="allow",
            config={k: str(v) for k, v in vars(args).items()} | dict(world_size=world),
        )

    def save_checkpoint(step):
        if not main_rank:
            return
        core.save_pretrained(checkpoint)
        torch.save(
            dict(step=step, optimiser=optimiser.state_dict(), report=report),
            checkpoint / "state.pt.tmp",
        )
        os.replace(checkpoint / "state.pt.tmp", checkpoint / "state.pt")
        # Keep every checkpointed adapter so the best held-out point can be picked.
        core.save_pretrained(args.out / f"adapter_s{step}")

    pad_id = processor.tokenizer.pad_token_id
    loader = iter(
        torch.utils.data.DataLoader(
            Examples(processor, tokenizer, train, args.chunk, args.photo_aug),
            batch_size=args.batch_size,
            sampler=ExampleOrder(len(train), 1155, (first - 1) * per_update, rank, world),
            num_workers=args.workers,
            collate_fn=Collate(pad_id),
            prefetch_factor=4 if args.workers else None,
            persistent_workers=bool(args.workers),
        )
    )
    # Each rank scores its own share of the fixed held-out sample.
    mine = held[rank::world]
    held_batches = [
        collate(
            [encode(processor, tokenizer, r, args.chunk) for r in mine[i : i + args.batch_size]],
            pad_id,
        )
        for i in range(0, len(mine), args.batch_size)
    ]
    # D169: simulator validation, batched per motion type so each type's loss is exact.
    # Every rank lists every type (possibly empty), so the reductions line up.
    sim_by_kind = {motion_kind(r["instruction"]): [] for r in sim_held}
    for r in sim_held[rank::world]:
        sim_by_kind[motion_kind(r["instruction"])].append(r)
    sim_batches = {
        kind: [
            collate(
                [encode(processor, tokenizer, r, args.chunk) for r in rows[i : i + args.batch_size]],
                pad_id,
            )
            for i in range(0, len(rows), args.batch_size)
        ]
        for kind, rows in sorted(sim_by_kind.items())
    }

    def scored(batch):
        """Summed loss, teacher-forced answer-token hits (exact and within one
        bin, OpenVLA's accuracy measure), answer tokens, examples (D170)."""
        batch = api.cuda(batch)
        out = core(**batch)
        target = batch["labels"][:, 1:]
        guess = out.logits[:, :-1].argmax(-1)
        # Accuracy over the move tokens (and end-of-answer) only: progress digits
        # (D175) are ordinary text ids, where "within one bin" means nothing.
        mask = (target != -100) & (target >= tokenizer.vocab_size - tokenizer.n_bins)
        count = batch["input_ids"].shape[0]
        return (
            float(out.loss) * count,
            int(((guess == target) & mask).sum()),
            int((((guess - target).abs() <= 1) & mask).sum()),
            int(mask.sum()),
            count,
        )

    def totals(batches):
        """Column sums of scored() over batches; zeros when a rank has none."""
        sums = [0.0, 0, 0, 0, 0]
        for batch in batches:
            sums = [total + part for total, part in zip(sums, scored(batch), strict=True)]
        return sums

    def reduced(value):
        if world == 1:
            return value
        tensor = torch.tensor([value], device="cuda", dtype=torch.float64)
        torch.distributed.all_reduce(tensor)
        return float(tensor)

    model.train()
    for step in range(first, args.updates + 1):
        assert time.monotonic() - start < args.max_wall_seconds, "wall clock budget exhausted"
        total = 0.0
        tick = time.monotonic()
        for group in optimiser.param_groups:
            group["lr"] = learning_rate(args, step, report)
        optimiser.zero_grad(set_to_none=True)
        for micro in range(args.accum):
            # Gradients only need averaging across GPUs on the last micro-batch.
            last = micro == args.accum - 1
            sync = contextlib.nullcontext() if world == 1 or last else model.no_sync()
            with sync:
                output = model(**api.cuda(next(loader)))
                loss = output.loss / args.accum
                assert torch.isfinite(loss), "non-finite loss"
                loss.backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
        total = reduced(total) / world
        report["losses"].append(round(total, 5))
        report["seconds_per_update"].append(round(time.monotonic() - tick, 3))
        if step % args.val_every == 0 or step == 1:
            core.eval()
            with torch.inference_mode():
                real = totals(held_batches)
                sim = {kind: totals(batches) for kind, batches in sim_batches.items()}
            core.train()
            real = [reduced(v) for v in real]
            held_loss = real[0] / len(held)
            entry = dict(
                step=step,
                loss=round(held_loss, 5),
                token_acc=round(real[1] / real[3], 4),
                token_acc_1bin=round(real[2] / real[3], 4),
            )
            logged = dict(
                held_out_loss=held_loss,
                held_out_token_acc=entry["token_acc"],
                held_out_token_acc_1bin=entry["token_acc_1bin"],
            )
            if sim:
                sim = {k: [reduced(x) for x in v] for k, v in sim.items()}
                count = sum(v[4] for v in sim.values())
                tokens = sum(v[3] for v in sim.values())
                entry["sim_loss"] = round(sum(v[0] for v in sim.values()) / count, 5)
                entry["sim_token_acc"] = round(sum(v[1] for v in sim.values()) / tokens, 4)
                entry["sim_token_acc_1bin"] = round(sum(v[2] for v in sim.values()) / tokens, 4)
                entry["sim_by_kind"] = {k: round(v[0] / v[4], 5) for k, v in sim.items()}
                entry["sim_acc_by_kind"] = {k: round(v[1] / v[3], 4) for k, v in sim.items()}
                logged["held_out_loss_sim"] = entry["sim_loss"]
                logged["held_out_token_acc_sim"] = entry["sim_token_acc"]
                logged["held_out_token_acc_1bin_sim"] = entry["sim_token_acc_1bin"]
                logged |= {f"sim_val/{k}": v for k, v in entry["sim_by_kind"].items()}
                logged |= {f"sim_acc/{k}": v for k, v in entry["sim_acc_by_kind"].items()}
            report["held_out"].append(entry)
            if args.schedule == "wsd" and report.get("decay_from") is None:
                # D177: with --plateau-sim the simulator accuracy counts too, so the
                # run goes flat only when neither real nor simulator improves.
                keys = ["token_acc", "token_acc_1bin"]
                if args.plateau_sim and "sim_token_acc" in entry:
                    keys += ["sim_token_acc", "sim_token_acc_1bin"]
                best = report.setdefault("real_best", {})
                history = report["held_out"][:-1]
                for key in keys:  # a resumed run starts from its history
                    if key not in best:
                        best[key] = max((e.get(key, -1.0) for e in history), default=-1.0)
                gained = False
                for key in keys:
                    if entry[key] > best[key]:
                        best[key], best[key + "_step"], gained = entry[key], step, True
                report["real_flat"] = 0 if gained else report.get("real_flat", 0) + 1
                if report["real_flat"] >= args.plateau_patience:
                    report["decay_from"], report["decay_reason"] = step, "accuracy flat"
                elif step >= args.updates - args.decay_updates:
                    report["decay_from"], report["decay_reason"] = step, "update cap"
                if report.get("decay_from") is not None and main_rank:
                    print(json.dumps(dict(decay_from=step, reason=report["decay_reason"])), flush=True)
                logged["real_flat_checks"] = report["real_flat"]
            if run:
                run.log(logged, step=step)
            # D169 early stop: watch the simulator loss (the test domain) when there
            # is one, else the real one; keep the best adapter, stop after
            # `patience` validations in a row without a new best.
            watched = entry.get("sim_loss", entry["loss"])
            if watched < report.get("best_loss", math.inf):
                report["best_loss"], report["best_step"], report["no_gain"] = watched, step, 0
                if main_rank and step > 1:
                    core.save_pretrained(args.out / "adapter_best")
            else:
                report["no_gain"] = report.get("no_gain", 0) + 1
            if args.early_stop_patience and report["no_gain"] >= args.early_stop_patience:
                report["stopped_early_at"] = step
                if main_rank:
                    print(json.dumps(dict(early_stop=step, best_step=report["best_step"])), flush=True)
                break
        if run:
            run.log(
                dict(
                    train_loss=total,
                    lr=optimiser.param_groups[0]["lr"],
                    seconds_per_update=report["seconds_per_update"][-1],
                    epoch=step * per_update / len(train),
                ),
                step=step,
            )
        if args.save_every and step % args.save_every == 0:
            save_checkpoint(step)
        if not main_rank:
            continue
        if step % 100 == 0:
            save()
            print(json.dumps(dict(step=step, loss=round(total, 4))), flush=True)
        elif step <= 10 or step % 10 == 0:
            seconds = report["seconds_per_update"][-1]
            print(json.dumps(dict(step=step, loss=round(total, 4), s=seconds)), flush=True)
        decay_from = report.get("decay_from")
        if args.schedule == "wsd" and decay_from is not None and step >= decay_from + args.decay_updates:
            report["finished_decay_at"] = step
            break

    report["status"] = "complete"
    report["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    if main_rank:
        core.save_pretrained(args.out / f"adapter_s{step}")
        save()
        print(json.dumps(dict(first_loss=report["losses"][0], last_loss=report["losses"][-1])))
    if run:
        run.finish()
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
