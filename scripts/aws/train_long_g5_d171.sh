#!/bin/bash
: "${VLA_BUCKET:?set VLA_BUCKET to your S3 bucket (see .env.example)}"
# D171 long run on the single-GPU Linux g5 (the P quota for a p4d is still
# pending). All 54 real shards + UAV-Flow-Sim are already in the store on this
# disk (D170). Continues from the D169b best adapter (update 2500) at the recipe
# learning rate 5e-4 with ONE warmup and a cosine decay over the whole run
# (docs/research/VLA_PLATEAU_20260929.md); no early stop on loss. Validation every
# 1,000 updates: loss and action-token accuracy, real (1,024) and simulator
# (1,024 from 504 held-out flights, per motion type). Adapters every 2,000
# updates, synced to S3 every hour. Stops the machine at the end; hard stop 50 h.
# If the p4d quota arrives, stop this run and continue from its last checkpoint.
set -x
LOG=/home/ubuntu/train_long_d171.log
exec >> $LOG 2>&1
sudo shutdown -h +3000 "D171 hard auto-stop"
BUCKET=s3://${VLA_BUCKET}/d171
R=/home/ubuntu/runs/official_k8_d171
export QWEN_MODEL=/home/ubuntu/qwen3vl4b UAV_FLOW_OFFICIAL_STORE=/home/ubuntu/data/uav_flow_official_10shard PYTHONPATH=/home/ubuntu/vla/src:/home/ubuntu/vla/scripts PYTHONWARNINGS=ignore
aws s3 cp s3://${VLA_BUCKET}/d170/train_uav_flow_vla.py /home/ubuntu/vla/scripts/train_uav_flow_vla.py
sed -i 's#reports/uav_flow_official_20260922#reports/uav_flow_official_10shard#' /home/ubuntu/vla/scripts/train_uav_flow_vla.py
wc -l $UAV_FLOW_OFFICIAL_STORE/episodes.jsonl; df -h / | tail -1
( while sleep 3600; do aws s3 sync $R $BUCKET/run/ --exclude "checkpoint/*" --exclude "wandb/*" --quiet; done ) &
cd /home/ubuntu/vla
/opt/pytorch/bin/python scripts/train_uav_flow_vla.py --format official --chunk 8 --precision bf16 --instruction both --mirror \
  --batch-size 8 --accum 4 --workers 6 --updates 26000 --lr 5e-4 --schedule cosine --warmup 0.03 \
  --val-every 1000 --val-batches 1024 --sim-val-per-kind 56 --sim-val-examples 1024 --save-every 2000 \
  --max-wall-seconds 172800 --init-adapter /home/ubuntu/runs/official_k8_d169b/adapter_best \
  --wandb-project "vla training" --out $R
echo "TRAIN_EXIT=$?"
aws s3 sync $R $BUCKET/run/ --exclude "checkpoint/*" --exclude "wandb/*" --quiet
aws s3 cp $LOG $BUCKET/train_long_d171.log
echo PIPELINE_DONE
sudo shutdown -h now
