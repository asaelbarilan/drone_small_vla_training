#!/bin/bash
: "${VLA_BUCKET:?set VLA_BUCKET to your S3 bucket (see .env.example)}"
# D193 box-first training on one Linux g5 (A10G 24 GB), user-approved 2026-10-05.
# Teaches the model to write the box of the instruction's object before its progress line,
# keeping everything learned so far (docs/PLAN.md "D193 checklist"):
#   - starts from the round-2 DAgger adapter (d190/dagger_r2_adapter: s31000 + both rounds);
#   - the full real + simulator store with the progress line, both wordings, mirror copies
#     (boxes mirrored with the photo);
#   - the DAgger rows of rounds 1 and 2, rebuilt here from their rollout zips (--extra-rows);
#   - photo augmentation for what the evaluator really sends (BGR swap, 224 px; D192);
#   - the same 504 held-out simulator flights for validation (rule 1: no test tasks).
# Steps: setup -> DAgger rows -> box labels (base Qwen, time-boxed) -> WAIT for the go file
# (s3://.../d193/go.txt: extra trainer flags decided by the D194 research; the code bundle is
# re-read then, so new trainer features can be added while labelling runs; no go within 3 h ->
# train as written) -> 30-update smoke -> training -> S3 -> machine off. Hard stop 22 h.
set -x
LOG=/home/ubuntu/train_box_d193.log
exec >> $LOG 2>&1
sudo shutdown -h +1320 "D193 hard auto-stop"
S3=s3://${VLA_BUCKET}
B=$S3/d193
D=/opt/dlami/nvme; [ -d $D ] || D=/home/ubuntu
sudo chown ubuntu:ubuntu $D
PY=/opt/pytorch/bin/python
STORE=$D/uav_flow_official_10shard
R=$D/runs/box_d193
export QWEN_MODEL=$D/qwen3vl4b UAV_FLOW_OFFICIAL_STORE=$STORE PYTHONPATH=/home/ubuntu/vla/src:/home/ubuntu/vla/scripts PYTHONWARNINGS=ignore
step() { echo "=== $1 $(date -u)"; aws s3 cp $LOG $B/train_box_d193.log --quiet; }
finish() { aws s3 sync $R $B/run/ --exclude "checkpoint/*" --quiet; aws s3 cp $LOG $B/train_box_d193.log; sudo shutdown -h now; exit; }

step "1 setup"
mkdir -p /home/ubuntu/vla && cd /home/ubuntu/vla
unpack() { aws s3 cp $B/vla_code.zip /home/ubuntu/vla_code.zip && $PY -c "import zipfile; zipfile.ZipFile('/home/ubuntu/vla_code.zip').extractall('/home/ubuntu/vla')"; }
unpack || finish
aws s3 cp $S3/d170/requirements_g5.txt /home/ubuntu/requirements_g5.txt
$PY -m pip install -q $(grep -iE '^(transformers|peft|accelerate)==' /home/ubuntu/requirements_g5.txt) huggingface_hub
$PY -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-VL-4B-Instruct', local_dir='$QWEN_MODEL')" || finish
aws s3 cp $S3/d170/store.tar - | tar -x -C $D || finish
ls $D; wc -l $STORE/episodes.jsonl || finish
aws s3 sync $S3/d190/dagger_r2_adapter/ $D/start_adapter/ --quiet
df -h $D | tail -1

step "2 DAgger rows (rounds 1 and 2)"
ROWS=""
for T in d188_dagger_r1 d190_dagger_r1; do
  aws s3 cp $S3/d172/$T.zip $D/$T.zip && $PY - "$D/$T.zip" "$D/$T" <<'EOF'
import sys, zipfile, pathlib
src, out = sys.argv[1], pathlib.Path(sys.argv[2])
with zipfile.ZipFile(src) as z:
    for i in z.infolist():
        name = i.filename.replace("\\", "/")
        if name.endswith("/"):
            continue
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(z.read(i))
EOF
  $PY scripts/rl/dagger_relabel.py --flights $D/$T/pass1 --calls $D/$T/server_calls.jsonl \
      --images $D/$T/rollout_images --tasks $D/$T/tasks --out $D/$T.rows.jsonl && ROWS="$ROWS $D/$T.rows.jsonl"
done
echo "DAgger rows: $ROWS"; wc -l $ROWS

step "3 box labels (base Qwen3-VL-4B, at most 3 h)"
$PY scripts/label_boxes.py --model $QWEN_MODEL --out $D/boxes.jsonl --every 8 --batch 16 --max-seconds 10800
aws s3 cp $D/boxes.jsonl $B/boxes.jsonl --quiet
$PY -c "import json; L=[json.loads(l) for l in open('$D/boxes.jsonl')]; print('labels', len(L), 'with box', sum(1 for x in L if x['box']))"

step "3b wait for go (up to 3 h)"
EXTRA=""
for i in $(seq 1 36); do
  if aws s3 cp $B/go.txt /home/ubuntu/go.txt --quiet; then EXTRA=$(cat /home/ubuntu/go.txt); unpack; break; fi
  sleep 300
done
echo "go flags: $EXTRA"

COMMON="--format official --chunk 8 --precision bf16 --instruction both --mirror --progress \
  --boxes $D/boxes.jsonl --box-repeat 3 --photo-aug --extra-rows $ROWS --extra-repeat 20 \
  --init-adapter $D/start_adapter --batch-size 8 --accum 4 --workers 6 \
  --sim-val-per-kind 56 --sim-val-examples 1024 $EXTRA"

step "4 smoke (30 updates)"
$PY scripts/train_uav_flow_vla.py $COMMON --updates 30 --lr 1e-4 --schedule constant \
  --val-every 30 --val-batches 64 --max-wall-seconds 3600 --out $D/runs/smoke_d193 || finish

step "5 training (4,000 updates, lr 1e-4, warmup 3 %, cosine to 0)"
( while sleep 3600; do aws s3 sync $R $B/run/ --exclude "checkpoint/*" --quiet; aws s3 cp $LOG $B/train_box_d193.log --quiet; done ) &
$PY scripts/train_uav_flow_vla.py $COMMON --updates 4000 --lr 1e-4 --schedule cosine --warmup 0.03 \
  --val-every 500 --val-batches 1024 --save-every 1000 --max-wall-seconds 57600 --out $R
echo "TRAIN_EXIT=$?"
step "6 done"
echo PIPELINE_DONE
finish
