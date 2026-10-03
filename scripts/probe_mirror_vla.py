"""D159: does our adapter's action change when the photo is mirrored?

The scale-free camera test, run on checkpoints during training so we can see
whether camera use appears at all. Mirroring is stronger than the D156 swap: the
scene is real, the geometry is the same, only left and right are exchanged, so a
model that uses the image must at least change its answer, and a model that
grounds "the tree on the left side" should flip its sideways command.

Reference point from the released OpenVLA-UAV on the same 20 held-out flights:
16 of 20 actions changed under mirroring, 6 flipped the sideways sign. Our D155
adapters changed in none.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PIL import ImageOps
from train_uav_flow_vla import (
    OFFICIAL_REPORT,
    collate_left,
    encode,
    load_image,
    official_rows,
    official_stats,
)
from train_uav_flow_vla import setup as load_model
from uav_flow_action_tokenizer import ActionTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter", nargs="+", required=True, help="one or more checkpoints; the base loads once"
    )
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--flights", type=int, default=20)
    parser.add_argument("--frames", type=int, default=3, help="frames per flight")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--precision", default="bf16", choices=["nf4", "bf16"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", type=Path, default=OFFICIAL_REPORT)
    args = parser.parse_args()

    by_flight = {}
    for row in official_rows("val", args.chunk):
        by_flight.setdefault(row["episode"], []).append(row)
    cases = []
    for flight in sorted(by_flight)[: args.flights]:
        steps = sorted(by_flight[flight], key=lambda r: r["step"])
        for index in np.linspace(0, len(steps) - 2, args.frames).astype(int):
            cases.append(steps[int(index)])

    api, processor, base, _ = load_model(args.model, args.precision)
    tokenizer = ActionTokenizer(processor.tokenizer, official_stats())
    pad_id = processor.tokenizer.pad_token_id

    # Encode once: the same real and mirrored inputs go to every checkpoint.
    encoded = []
    for row in cases:
        image = load_image(row)
        for picture in (image, ImageOps.mirror(image)):
            encoded.append(
                encode(processor, tokenizer, row, args.chunk, with_answer=False, image=picture)
            )

    model = None
    summaries = []
    for number, adapter in enumerate(args.adapter):
        name = f"a{number}"
        if model is None:
            model = PeftModel.from_pretrained(base, adapter, adapter_name=name)
        else:
            model.load_adapter(adapter, adapter_name=name)
        model.set_adapter(name)
        model.eval()

        predictions = []
        for start in range(0, len(encoded), args.batch_size):
            group = encoded[start : start + args.batch_size]
            batch = collate_left(group, pad_id)
            width = batch["input_ids"].shape[1]
            with torch.inference_mode():
                out = model.generate(
                    **api.cuda(batch),
                    max_new_tokens=4 * args.chunk + 2,
                    do_sample=False,
                    use_cache=True,
                )
            predictions.extend(
                tokenizer.decode(out[i, width:].tolist(), args.chunk) for i in range(len(group))
            )

        records, changed, flipped_y, flipped_yaw, unparsed = [], 0, 0, 0, 0
        for index, row in enumerate(cases):
            real, mirrored = predictions[2 * index], predictions[2 * index + 1]
            if real is None or mirrored is None:
                unparsed += 1
                continue
            changed += int(not np.allclose(real, mirrored))
            flipped_y += int(
                np.sign(real[0][1]) != 0 and np.sign(mirrored[0][1]) == -np.sign(real[0][1])
            )
            flipped_yaw += int(
                np.sign(real[0][3]) != 0 and np.sign(mirrored[0][3]) == -np.sign(real[0][3])
            )
            records.append(
                dict(
                    episode=row["episode"],
                    step=row["step"],
                    instruction=row["instruction"],
                    real=np.round(real[0], 4).tolist(),
                    mirrored=np.round(mirrored[0], 4).tolist(),
                )
            )
        total = len(records)
        summary = dict(
            adapter=adapter,
            chunk=args.chunk,
            comparisons=total,
            unparsed=unparsed,
            changed_when_mirrored=changed,
            changed_rate=round(changed / total, 3) if total else None,
            sideways_sign_flipped=flipped_y,
            yaw_sign_flipped=flipped_yaw,
            reference_openvla_uav="16/20 changed, 6/20 sideways sign flipped (first frames)",
        )
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"mirror_{args.tag}_{Path(adapter).name}.json").write_text(
            json.dumps(dict(summary=summary, records=records), indent=2), encoding="utf-8"
        )
        if number > 0:
            model.delete_adapter(name)  # keep memory flat across many checkpoints

    (args.out_dir / f"mirror_{args.tag}_trend.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
