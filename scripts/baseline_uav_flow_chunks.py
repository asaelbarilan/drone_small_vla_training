"""D155: text-only baselines for the chunked UAV-Flow splits.

The D153 baseline averaged training episodes sharing a validation episode's exact
instruction. On `split_unseen` that is impossible by construction - zero of 62
validation instructions appear in training - so it degenerates to the global mean
and would understate what text alone can do.

The honest competitor on an unseen split is a model that generalises across
wording, so the text baseline here is a nearest-neighbour over instruction tokens:
TF-IDF cosine against every training instruction, take the k closest, average their
endpoints. It never sees an image.

Both splits are scored. `split_seen` is reported only so the seen-to-unseen gap
can be read off; the literature's gap is severe (LongFly 36.39 to 11.27) and ours
should be stated rather than discovered later.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/uav_flow_chunks_20260921"
STORE = Path("D:/drone_vla_pilot/data/uav_flow_chunks_20260921")
NEIGHBOURS = 5
WORD = re.compile(r"[a-z]+")


def tokens(text):
    return WORD.findall(text.lower())


def build_vectors(instructions):
    """TF-IDF over instruction words, L2 normalised, as plain dicts."""
    documents = [Counter(tokens(t)) for t in instructions]
    frequency = Counter()
    for d in documents:
        frequency.update(d.keys())
    n = len(documents)
    vectors = []
    for d in documents:
        v = {w: c * math.log((n + 1) / (frequency[w] + 1)) for w, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vectors.append({w: x / norm for w, x in v.items()})
    return vectors


def cosine(a, b):
    small, large = (a, b) if len(a) < len(b) else (b, a)
    return sum(x * large.get(w, 0.0) for w, x in small.items())


def score(split, episodes):
    train = [e for e in episodes if e[split] == "train"]
    val = [e for e in episodes if e[split] == "val"]
    if not train or not val:
        return None

    train_end = np.array([e["endpoint_m"] for e in train])
    overall = train_end.mean(axis=0)
    train_vectors = build_vectors([e["instruction"] for e in train])
    val_vectors = build_vectors([e["instruction"] for e in val])

    exact = {e["instruction"].lower() for e in train}
    global_errors, neighbour_errors, similarities = [], [], []
    for row, vector in zip(val, val_vectors, strict=True):
        truth = np.array(row["endpoint_m"])
        global_errors.append(float(np.linalg.norm(truth - overall)))
        sims = np.array([cosine(vector, t) for t in train_vectors])
        top = np.argsort(sims)[-NEIGHBOURS:]
        similarities.append(float(sims[top[-1]]))
        neighbour_errors.append(float(np.linalg.norm(truth - train_end[top].mean(axis=0))))

    median = lambda v: round(float(np.median(v)), 3)  # noqa: E731
    return dict(
        train_episodes=len(train),
        val_episodes=len(val),
        val_instructions_with_exact_train_match=sum(e["instruction"].lower() in exact for e in val),
        median_trajectory_m=median([np.linalg.norm(e["endpoint_m"]) for e in val]),
        no_text=dict(median_final_error_m=median(global_errors)),
        text_nearest_neighbour=dict(
            k=NEIGHBOURS,
            median_final_error_m=median(neighbour_errors),
            median_top1_similarity=median(similarities),
        ),
        target_to_beat_m=min(median(global_errors), median(neighbour_errors)),
    )


def main():
    episodes = [
        json.loads(s) for s in (STORE / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    report = dict(
        neighbours=NEIGHBOURS,
        note="text baselines never see an image; the bar is the lower of the two",
        split_unseen=score("split_unseen", episodes),
        split_seen=score("split_seen", episodes),
    )
    gap = report["split_unseen"]["target_to_beat_m"] - report["split_seen"]["target_to_beat_m"]
    report["baseline_gap_unseen_minus_seen_m"] = round(gap, 3)
    (OUT / "baselines.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
