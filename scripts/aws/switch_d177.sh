#!/bin/bash
: "${VLA_BUCKET:?set VLA_BUCKET to your S3 bucket (see .env.example)}"
# D177: same run, new stop rule. At the next 2,000-update checkpoint (26,000, or
# 28,000 if 26,000 is past) stop the D176 script and trainer, take the trainer with
# --plateau-sim, and resume the SAME run with the same learning rate, progress
# tokens on, NO plateau reset: the run now goes flat only when neither real nor
# simulator accuracy makes a new best for 3 validations. The D176 hard stop
# (2026-10-05 about 03:35 UTC) stays in place. Machine off at the end.
set -x
LOG=/home/ubuntu/switch_d177.log
exec >> $LOG 2>&1
R=/home/ubuntu/runs/official_k8_d171
BUCKET=s3://${VLA_BUCKET}
PY=/opt/pytorch/bin/python
TARGET=26000
[ -d $R/adapter_s26000 ] && TARGET=28000
echo "waiting for adapter_s$TARGET $(date -u)"
until [ -d $R/adapter_s$TARGET ]; do sleep 60; done
sleep 90   # adapter_s<step> is written after checkpoint/state.pt
$PY -c "import torch; print('checkpoint step', torch.load('$R/checkpoint/state.pt', weights_only=False)['step'])"

pkill -f "bash /home/ubuntu/switch_d176.sh"; sleep 2
pkill -f train_uav_flow_vla.py
while pgrep -f "[t]rain_uav_flow_vla.py" > /dev/null; do sleep 5; done
echo "D176 stopped at $(date -u)"
cat /run/systemd/shutdown/scheduled

aws s3 cp $BUCKET/d177/train_uav_flow_vla.py /home/ubuntu/vla/scripts/train_uav_flow_vla.py
grep -c "plateau-sim" /home/ubuntu/vla/scripts/train_uav_flow_vla.py

export QWEN_MODEL=/home/ubuntu/qwen3vl4b UAV_FLOW_OFFICIAL_STORE=/home/ubuntu/data/uav_flow_official_10shard PYTHONPATH=/home/ubuntu/vla/src:/home/ubuntu/vla/scripts PYTHONWARNINGS=ignore
( while sleep 3600; do aws s3 sync $R $BUCKET/d171/run/ --exclude "checkpoint/*" --exclude "wandb/*" --quiet; done ) &
SYNC=$!
( sleep 600; pgrep -f "[w]atch_d171.py" > /dev/null || nohup $PY /home/ubuntu/watch_d171.py > /dev/null 2>&1 ) &
cd /home/ubuntu/vla
$PY scripts/train_uav_flow_vla.py --format official --chunk 8 --precision bf16 --instruction both --mirror \
  --batch-size 8 --accum 4 --workers 6 --updates 56000 --lr 1.1420e-04 --schedule wsd --warmup 0 \
  --plateau-patience 3 --decay-updates 3000 --plateau-sim \
  --val-every 1000 --val-batches 1024 --sim-val-per-kind 56 --sim-val-examples 1024 --save-every 2000 \
  --max-wall-seconds 250000 --init-adapter /home/ubuntu/runs/official_k8_d169b/adapter_best \
  --wandb-project "vla training" --out $R --resume --progress
echo "TRAIN_EXIT=$?"
kill $SYNC
aws s3 sync $R $BUCKET/d171/run/ --exclude "checkpoint/*" --exclude "wandb/*" --quiet
aws s3 cp $LOG $BUCKET/d177/switch_d177.log
aws s3 cp /home/ubuntu/switch_d176.log $BUCKET/d176/switch_d176.log
echo PIPELINE_DONE
sudo shutdown -h now
