# D153: UAV-Flow endpoint pilot

## Why a new dataset

D152 showed the OpenFly panel could not rank models: an oracle ignoring every
image scored 36/72 by route identity alone, above all four models. The user also
objected that the task shape was wrong — the prompt enumerated the answer set
("return exactly one digit: 0=stop, 1=forward 3 m…"), which is not how these
models are used. D149's own audit agrees: most drone VLAs emit velocities or
waypoints, and OpenFly learns action tokens rather than being handed a menu.

UAV-Flow was chosen because three separate papers report on it (WorldVLN 79.12%,
ImagineUAV 70.9%, FLIGHTVLA 59.0%), its instructions are single free-form goals,
and its actions are continuous real-flight trajectories. One 4.7 GB shard of 54
was downloaded. **The dataset declares no license**; that is unresolved and must
be settled before any result built on it is published.

## Task and frozen split

First frame plus instruction → final displacement (x, y, z) in metres in the
start frame, emitted as three numbers. No answer set appears in the prompt.

500 episodes, split by hash of the episode id: 412 train / 88 validation. 76 of
the 88 validation episodes share their instruction with a training episode.
Median validation trajectory length is 7.34 m.

## Baselines, computed before training

Predicting each validation endpoint without ever seeing an image, using training
episodes only:

| predictor | median final error |
| --- | --- |
| global mean, no text at all | 5.79 m |
| mean of training episodes with the same instruction | **2.26 m** |

The second number is the bar. A model that cannot beat it is not earning its
vision.

## Result: SmolVLM-256M + LoRA, 400 updates

| condition | median final error | distinct predictions |
| --- | --- | --- |
| model, real frames | **3.21 m** | 56 / 88 |
| model, blinded (flat gray frames) | 6.47 m | 6 / 88 |
| text-only baseline | 2.26 m | — |
| no-text baseline | 5.79 m | — |

Paired sign tests over the 88 validation episodes:

- model with vision versus blinded: better on **62 of 83**, p = 7.5e-06
- model with vision versus text-only baseline: better on **30 of 88**, p = 0.0037

## What this shows

**The task format works.** This is the first model in this project whose score
demonstrably depends on the camera. Blinding it doubles the error and collapses
it from 56 distinct answers to 6. The OpenFly adapters showed no such effect.

**The model is not yet good enough.** It loses to a trivial instruction lookup,
significantly. With 412 training episodes and 400 updates on a 256M model this is
a scale result, not a verdict on the approach, but it is a real loss and is not
to be reported as a success.

The next lever is data — additional shards at roughly 500 episodes each — not
more updates on 412 episodes.

## Harness defects found and fixed

1. The first parser searched anywhere in the output and scraped digits out of
   prompt text the untrained model echoed back, scoring that as a 1049 m
   prediction rather than a failure. Replaced with an anchored match.
2. The replacement required an exact full match, and the model emits a trailing
   fourth number because no end-of-sequence token was ever trained, so all 88
   predictions were discarded as unparsed. The scoring pass anchors to the start
   and takes the first three numbers; the training script now trains an EOS token.
3. The manifest builder hashed a shard prefix via `read_bytes()`, pulling 4.6 GB
   into memory. Fixed to read one megabyte.

The reported numbers come from `score_uav_flow_pilot.py` re-running the saved
adapter over the full validation split, not from the training loop's own counters.

## Limits

Endpoint only, not the full trajectory. One shard of 54. One model, one seed, one
hyperparameter setting. No flight execution and no comparison to any published
UAV-Flow number — those papers score executed flights, which this does not.

## Evidence

`manifest.json` (split, hashes, baselines), `scored_s400.json` (per-episode
predictions under both conditions), and the run at
`D:/drone_vla_pilot/runs/uav_flow_pilot_20260921_a/report.json`.

Reproduction: `prepare_uav_flow_pilot.py`, `train_uav_flow_pilot.py`,
`score_uav_flow_pilot.py`.
