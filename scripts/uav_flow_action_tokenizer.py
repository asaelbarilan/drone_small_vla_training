"""D155: OpenVLA's action tokenizer, ported to the SmolVLM vocabulary.

The official UAV-Flow recipe (OpenVLA-UAV/prismatic/vla/action_tokenizer.py)
discretizes each action dimension into 256 uniform bins over [-1, 1] and maps them
onto the last 256 tokens of the vocabulary, so one dimension costs one token. Our
first attempt emitted free-text decimals at roughly five tokens per number, which
made an 8-step chunk ~250 tokens and forced a fragile parser - the source of both
harness bugs in D153.

Verified before adopting: SmolVLM's vocab_size is 49152, and the 256 ids below it
hold no special or added tokens and never appear in our prompts. OpenVLA's
assumption about the vocabulary tail therefore transfers.

Normalisation follows the same convention: per-channel q01/q99 over the training
split, mapped to [-1, 1] and clipped. Those statistics are this dataset's
equivalent of the recipe's `unnorm_key`, and are written to action_stats.json so
decoding can never silently use different numbers than encoding.
"""

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/uav_flow_chunks_20260921"
STORE = Path(os.environ.get("UAV_FLOW_STORE", "D:/drone_vla_pilot/data/uav_flow_chunks_20260921"))
BINS = 256
CHANNELS = 6


class ActionTokenizer:
    """One token per action dimension, following the official implementation."""

    def __init__(self, tokenizer, stats, bins=BINS):
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.low = np.asarray(stats["q01"], dtype=float)
        self.high = np.asarray(stats["q99"], dtype=float)
        self.span = np.where(self.high - self.low == 0, 1.0, self.high - self.low)
        self.bins = np.linspace(-1.0, 1.0, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = tokenizer.vocab_size
        # 6 for the D155 start-frame format, 4 for the official (dx, dy, dz, dyaw).
        self.channels = len(self.low)

    def normalise(self, action):
        return np.clip(
            2.0 * (np.asarray(action, dtype=float) - self.low) / self.span - 1.0, -1.0, 1.0
        )

    def denormalise(self, normalised):
        return (np.asarray(normalised, dtype=float) + 1.0) / 2.0 * self.span + self.low

    def token_ids(self, action):
        """Action rows (n, 6) in real units -> flat list of token ids."""
        normalised = self.normalise(action)
        discretized = np.digitize(normalised, self.bins)
        return (self.vocab_size - discretized).reshape(-1).tolist()

    def encode(self, action):
        return self.tokenizer.decode(self.token_ids(action))

    def decode(self, token_ids, steps):
        """Token ids -> (steps, channels) actions in real units, or None if malformed."""
        ids = np.asarray([i for i in token_ids if self.is_action_token(i)], dtype=int)
        if ids.size < steps * self.channels:
            return None
        ids = ids[: steps * self.channels]
        discretized = np.clip(self.vocab_size - ids - 1, 0, self.bin_centers.shape[0] - 1)
        return self.denormalise(self.bin_centers[discretized].reshape(steps, self.channels))

    def is_action_token(self, token_id):
        return self.vocab_size - self.n_bins <= token_id < self.vocab_size


def compute_stats(split="split_unseen"):
    rows = (STORE / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    values = []
    for line in rows:
        row = json.loads(line)
        if row[split] == "train":
            values.extend(row["chunk"])
    array = np.asarray(values, dtype=float)
    return dict(
        split=split,
        steps=int(array.shape[0]),
        q01=[round(float(v), 5) for v in np.percentile(array, 1, axis=0)],
        q99=[round(float(v), 5) for v in np.percentile(array, 99, axis=0)],
        median=[round(float(v), 5) for v in np.median(array, axis=0)],
        note="per-channel q01/q99 over training chunks; the unnorm_key equivalent",
    )


def load_stats(split="split_unseen"):
    return json.loads((REPORT / "action_stats.json").read_text(encoding="utf-8"))[split]


if __name__ == "__main__":
    stats = {s: compute_stats(s) for s in ("split_unseen", "split_seen")}
    (REPORT / "action_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
