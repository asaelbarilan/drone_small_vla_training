#!/bin/bash
: "${VLA_BUCKET:?set VLA_BUCKET to your S3 bucket (see .env.example)}"
# D179: user decision - end the run with the decay. At the 28,000 checkpoint, stops the D177 script and
# trainer, resumes from that checkpoint with --decay-now: cosine from
# the steady learning rate 1.142e-4 to 0 over 3,000 updates (progress tokens on),
# then the final adapter (adapter_s31000), S3 sync and machine off. The few updates
# after 28,000 that the stopped process made in the 90 s wait are not kept.
set -x
LOG=/home/ubuntu/decay_d179.log
exec >> $LOG 2>&1
R=/home/ubuntu/runs/official_k8_d171
BUCKET=s3://${VLA_BUCKET}
PY=/opt/pytorch/bin/python
# User: wait for the 28,000 checkpoint so no updates are thrown away (decay 28k -> 31k).
echo "waiting for adapter_s28000 $(date -u)"
until [ -d $R/adapter_s28000 ]; do sleep 60; done
sleep 90   # adapter_s<step> is written after checkpoint/state.pt
pkill -f "bash /home/ubuntu/switch_d177.sh"; sleep 2
pkill -f train_uav_flow_vla.py
while pgrep -f "[t]rain_uav_flow_vla.py" > /dev/null; do sleep 5; done
echo "D177 stopped at $(date -u)"
$PY -c "import torch; print('checkpoint step', torch.load('$R/checkpoint/state.pt', weights_only=False)['step'])"
sudo shutdown -c; sudo shutdown -h +540 "D179 hard auto-stop"
cat /run/systemd/shutdown/scheduled

aws s3 cp $BUCKET/d179/train_uav_flow_vla.py /home/ubuntu/vla/scripts/train_uav_flow_vla.py
grep -c "decay-now" /home/ubuntu/vla/scripts/train_uav_flow_vla.py

export QWEN_MODEL=/home/ubuntu/qwen3vl4b UAV_FLOW_OFFICIAL_STORE=/home/ubuntu/data/uav_flow_official_10shard PYTHONPATH=/home/ubuntu/vla/src:/home/ubuntu/vla/scripts PYTHONWARNINGS=ignore
( while sleep 3600; do aws s3 sync $R $BUCKET/d171/run/ --exclude "checkpoint/*" --exclude "wandb/*" --quiet; done ) &
SYNC=$!
cd /home/ubuntu/vla
$PY scripts/train_uav_flow_vla.py --format official --chunk 8 --precision bf16 --instruction both --mirror \
  --batch-size 8 --accum 4 --workers 6 --updates 56000 --lr 1.1420e-04 --schedule wsd --warmup 0 \
  --plateau-patience 3 --decay-updates 3000 --plateau-sim --decay-now \
  --val-every 1000 --val-batches 1024 --sim-val-per-kind 56 --sim-val-examples 1024 --save-every 2000 \
  --max-wall-seconds 40000 --init-adapter /home/ubuntu/runs/official_k8_d169b/adapter_best \
  --wandb-project "vla training" --out $R --resume --progress
echo "TRAIN_EXIT=$?"
kill $SYNC
aws s3 sync $R $BUCKET/d171/run/ --exclude "checkpoint/*" --exclude "wandb/*" --quiet
aws s3 cp $LOG $BUCKET/d179/decay_d179.log
aws s3 cp /home/ubuntu/switch_d177.log $BUCKET/d177/switch_d177.log
echo PIPELINE_DONE
sudo shutdown -h now
