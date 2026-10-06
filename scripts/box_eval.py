"""D198: does the model really find the target? Generated-box checks (the loss could not show it).

D193 trained the box format but the model wrote nearly the same box on every photo, and its
progress line ignored the box; teacher-forced loss and move-token accuracy both looked fine.
These checks let the model WRITE its answer and compare it with the labels:

- centre_err: mean distance between written and labelled box centres (0-1000 frame; lower is
  better). A model that writes the average box scores about the labels' own spread.
- iou: mean IoU of written vs labelled box (0 when one of them is empty).
- x_sd_ratio: spread of the written box centres / spread of the labelled ones. Near 0 = one box
  for every photo. Spread alone is not enough (D198: the D193 model's boxes spread like the
  labels but sat in the wrong places), so also:
- x_corr: correlation of written vs labelled box centre x (1 = follows the target left/right).
- centre_err_const: the centre error of a dumb "always the image centre" box on the same rows -
  the baseline the model must beat.
- empty_acc: written box empty exactly when the label is empty.
- next_centre_err: the same for the box after the moves (imagination), when labelled.
- turn_consistency (rows with a progress line wanted): on Turn-to-object rows whose target is
  clearly off-centre, the share where the line turns toward the side the WRITTEN box is on.
"""

import math
import re

import torch

from train_uav_flow_vla import append_ids, encode, parse_box, parse_progress

PREFIX = '{"bbox_2d": ['
TURN_TO_OBJECT = re.compile(r"^(turn|face)\b.*\b(dog|person|pet|animal|human)", re.IGNORECASE)


def _centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _sd(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


@torch.inference_mode()
def evaluate_boxes(model, processor, tokenizer, rows, k, cuda, want_line=False, image_fn=None):
    """Generate the box answer for each row (greedy, after the forced box prefix) and score it.
    `rows` must carry a `box` label (list, maybe empty) and may carry `next_box`.
    `image_fn(row)` overrides how the photo is loaded (local checks)."""
    new_tokens = 110 if want_line else 52
    prefix = processor.tokenizer.encode(PREFIX, add_special_tokens=False)
    errs, ious, empty_hits, next_errs, pred_x, label_x, turn_hits, turn_n = [], [], [], [], [], [], 0, 0
    const_errs = []
    for row in rows:
        query = {key: v for key, v in row.items() if key not in ("box", "next_box", "progress")}
        image = image_fn(row) if image_fn else None
        batch = append_ids(encode(processor, tokenizer, query, k, with_answer=False, image=image), prefix)
        ids = model.generate(
            **cuda(batch), max_new_tokens=new_tokens, do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )[0, batch["input_ids"].shape[1] :].tolist()
        text = PREFIX + processor.tokenizer.decode(ids, skip_special_tokens=True)
        head = text.split("Left")[0]
        written = parse_box(head) or []
        label = row["box"]
        empty_hits.append(bool(written) == bool(label))
        if label and written:
            (wx, wy), (lx, ly) = _centre(written), _centre(label)
            errs.append(math.hypot(wx - lx, wy - ly))
            const_errs.append(math.hypot(500 - lx, 500 - ly))
            pred_x.append(wx)
            label_x.append(lx)
        ious.append(_iou(written, label))
        after = head.split("bbox_2d_next", 1)
        written_next = parse_box('"bbox_2d' + after[1]) if len(after) == 2 else None
        if row.get("next_box") and written_next:
            (wx, wy), (lx, ly) = _centre(written_next), _centre(row["next_box"])
            next_errs.append(math.hypot(wx - lx, wy - ly))
        if want_line and written and TURN_TO_OBJECT.search(row.get("instruction", "")):
            offset = _centre(written)[0] - 500
            line = parse_progress(text)
            if abs(offset) > 100 and line is not None:
                turn_n += 1
                turn_hits += int((line[3] > 0) == (offset > 0) and line[3] != 0)
    mean = lambda v: round(sum(v) / len(v), 4) if v else None  # noqa: E731
    label_sd = _sd(label_x)
    corr = None
    if len(pred_x) > 2 and _sd(pred_x) and label_sd:
        mp, ml = sum(pred_x) / len(pred_x), sum(label_x) / len(label_x)
        cov = sum((a - mp) * (b - ml) for a, b in zip(pred_x, label_x)) / (len(pred_x) - 1)
        corr = round(cov / (_sd(pred_x) * label_sd), 3)
    return dict(
        box_rows=len(rows),
        box_centre_err=mean(errs),
        box_iou=mean(ious),
        box_x_sd_ratio=round(_sd(pred_x) / label_sd, 3) if label_sd else None,
        box_x_corr=corr,
        centre_err_const=mean(const_errs),
        box_empty_acc=mean([float(h) for h in empty_hits]),
        next_centre_err=mean(next_errs),
        turn_consistency=round(turn_hits / turn_n, 3) if turn_n else None,
        turn_rows=turn_n,
    )


def balanced_box_rows(rows, n, seed=7):
    """Up to n rows with a box label, as many left / centre / right targets as possible
    (and a few empty boxes), so an "average box" cannot look good."""
    import random

    bins = {"left": [], "centre": [], "right": [], "empty": []}
    seen = set()
    for r in rows:
        if "box" not in r or r["id"] in seen:
            continue
        seen.add(r["id"])
        if not r["box"]:
            bins["empty"].append(r)
        else:
            x = _centre(r["box"])[0]
            bins["left" if x < 400 else "right" if x > 600 else "centre"].append(r)
    picker = random.Random(seed)
    share = {"left": 0.3, "centre": 0.3, "right": 0.3, "empty": 0.1}
    out = []
    for name, part in share.items():
        group = bins[name]
        out += picker.sample(group, min(len(group), round(n * part)))
    return out


def balanced_sample(rows, size, seed=7):
    """Exactly `size` training rows with a box label: 30 % left, 30 % centre, 30 % right targets,
    10 % empty boxes, each bin cycled (repeated) when it is smaller than its share."""
    import itertools
    import random

    bins = {"left": [], "centre": [], "right": [], "empty": []}
    seen = set()
    for r in rows:
        if "box" not in r or r["id"] in seen:
            continue
        seen.add(r["id"])
        if not r["box"]:
            bins["empty"].append(r)
        else:
            x = _centre(r["box"])[0]
            bins["left" if x < 400 else "right" if x > 600 else "centre"].append(r)
    picker = random.Random(seed)
    share = {"left": 0.3, "centre": 0.3, "right": 0.3, "empty": 0.1}
    live = {k: v for k, v in bins.items() if v}
    total = sum(share[k] for k in live)
    out = []
    for name, group in live.items():
        picker.shuffle(group)
        out += list(itertools.islice(itertools.cycle(group), round(size * share[name] / total)))
    picker.shuffle(out)
    return out[:size]
