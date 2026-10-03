#!/bin/bash
: "${VLA_BUCKET:?set VLA_BUCKET to your S3 bucket (see .env.example)}"
# D176: switch the running D171 job to "train until real accuracy stops improving".
# 1. wait for the next 2,000-update checkpoint (16,000, or 18,000 if 16,000 is past);
# 2. stop the D171 script (so it does not shut the machine down), its hourly sync
#    and the trainer; cancel the D171 hard stop;
# 3. resume the SAME run (same output folder, optimiser state, data stream, wandb
#    run) with the new trainer: schedule wsd at the learning rate D171 had at that
#    step (no new warm-up), real-accuracy plateau patience 3 validations, then a
#    3,000-update cosine decay; cap 56,000 updates (about 69 h from 16,000);
# 4. hourly S3 sync + the simulator-accuracy watcher again; machine off at the end.
# Merged option (user, 2026-10-01): progress tokens (D175) switch on at the resume.
# Before that, a 30-update --progress smoke from the checkpoint adapter and the
# server smoke (greedy + goal memory, sampling); if either fails the run resumes
# WITHOUT progress tokens rather than stopping.
set -x
LOG=/home/ubuntu/switch_d176.log
exec >> $LOG 2>&1
R=/home/ubuntu/runs/official_k8_d171
BUCKET=s3://${VLA_BUCKET}
PY=/opt/pytorch/bin/python
TARGET=16000
[ -d $R/adapter_s16000 ] && TARGET=18000
echo "waiting for adapter_s$TARGET $(date -u)"
until [ -d $R/adapter_s$TARGET ]; do sleep 60; done
sleep 90   # adapter_s<step> is written after checkpoint/state.pt
$PY -c "import torch; print('checkpoint step', torch.load('$R/checkpoint/state.pt', weights_only=False)['step'])"

pkill -f train_long_g5_d171.sh; sleep 2
pkill -f train_uav_flow_vla.py
while pgrep -f "[t]rain_uav_flow_vla.py" > /dev/null; do sleep 5; done
sudo shutdown -c
sudo shutdown -h +4620 "D176 hard auto-stop"
echo "D171 stopped at $(date -u)"; nvidia-smi --query-gpu=memory.used --format=csv

aws s3 cp $BUCKET/d176/vla_scripts.zip /home/ubuntu/vla_scripts.zip
cd /home/ubuntu/vla && $PY -m zipfile -e /home/ubuntu/vla_scripts.zip . && cd /home/ubuntu
grep -n "uav_flow_official_10shard\|\"wsd\"\|reset-plateau" /home/ubuntu/vla/scripts/train_uav_flow_vla.py | head -5
$PY -c "import flask" 2>/dev/null || $PY -m pip install -q flask
LR=$($PY -c "
import math
step, warm, total = $TARGET, round(0.03 * 26000), 26000
print('%.4e' % (5e-4 * 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))))")
echo "resume lr $LR"

export QWEN_MODEL=/home/ubuntu/qwen3vl4b UAV_FLOW_OFFICIAL_STORE=/home/ubuntu/data/uav_flow_official_10shard PYTHONPATH=/home/ubuntu/vla/src:/home/ubuntu/vla/scripts PYTHONWARNINGS=ignore
( while sleep 3600; do aws s3 sync $R $BUCKET/d171/run/ --exclude "checkpoint/*" --exclude "wandb/*" --quiet; done ) &
SYNC=$!
cd /home/ubuntu/vla
S=/home/ubuntu/runs/d176_smoke
FLAG="--progress --reset-plateau"
$PY scripts/train_uav_flow_vla.py --format official --chunk 8 --precision bf16 --instruction both --mirror   --batch-size 8 --accum 4 --workers 6 --updates 30 --lr $LR --schedule constant --val-every 30 --val-batches 64   --progress --init-adapter $R/adapter_s$TARGET --max-wall-seconds 3600 --out $S   && $PY scripts/smoke_progress_server.py --adapter $S/adapter_s30 --goal-memory --out $S/server_greedy   && $PY scripts/smoke_progress_server.py --adapter $S/adapter_s30 --temperature 1.0 --out $S/server_sample   && echo SMOKE_OK || { echo "SMOKE_FAILED - resuming without progress tokens"; FLAG=""; }
aws s3 sync $S $BUCKET/d176/smoke/ --quiet
( sleep 600; pgrep -f "[w]atch_d171.py" > /dev/null || nohup $PY /home/ubuntu/watch_d171.py > /dev/null 2>&1 ) &
$PY scripts/train_uav_flow_vla.py --format official --chunk 8 --precision bf16 --instruction both --mirror \
  --batch-size 8 --accum 4 --workers 6 --updates 56000 --lr $LR --schedule wsd --warmup 0 \
  --plateau-patience 3 --decay-updates 3000 \
  --val-every 1000 --val-batches 1024 --sim-val-per-kind 56 --sim-val-examples 1024 --save-every 2000 \
  --max-wall-seconds 270000 --init-adapter /home/ubuntu/runs/official_k8_d169b/adapter_best \
  --wandb-project "vla training" --out $R --resume $FLAG
echo "TRAIN_EXIT=$?"
kill $SYNC
aws s3 sync $R $BUCKET/d171/run/ --exclude "checkpoint/*" --exclude "wandb/*" --quiet
aws s3 cp $LOG $BUCKET/d176/switch_d176.log
aws s3 cp /home/ubuntu/train_long_d171.log $BUCKET/d171/train_long_d171.log
echo PIPELINE_DONE
sudo shutdown -h now
