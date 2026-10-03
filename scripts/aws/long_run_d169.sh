#!/bin/bash
: "${VLA_BUCKET:?set VLA_BUCKET to your S3 bucket (see .env.example)}"
# D169 on the Linux g5: long real+sim training with simulator validation -> S3 -> stop.
# Continues the D165 adapter (real+sim s2000) for 10,000 updates. 504 simulator
# flights (56 per motion type) are held out for validation; real validation grows
# from 128 to 1,024 examples. Uploads every adapter, the report and the log, then
# shuts the machine down, also when training fails. Hard stop after 20 h.
# Early stop: after 3 validations in a row (every 500 updates) without a new
# best simulator validation loss; adapter_best keeps the best point.
# lr 5e-5 (was 2e-4): at 2e-4 validation rose at update 500 (sim 0.737 -> 0.767),
# the same high-rate bump as D165, which early stop would have cut off (D169b).
set -x
LOG=/home/ubuntu/long_run_d169.log
exec >> $LOG 2>&1
sudo shutdown -h +1200 "D169 hard auto-stop"
BUCKET=s3://${VLA_BUCKET}/d169
export QWEN_MODEL=/home/ubuntu/qwen3vl4b UAV_FLOW_OFFICIAL_STORE=/home/ubuntu/data/uav_flow_official_10shard PYTHONPATH=/home/ubuntu/vla/src:/home/ubuntu/vla/scripts PYTHONWARNINGS=ignore
aws s3 cp $BUCKET/train_uav_flow_vla.py /home/ubuntu/vla/scripts/train_uav_flow_vla.py
# The instance keeps the 10-shard manifest under this name (see AGENTS.md D159).
sed -i 's#reports/uav_flow_official_20260922#reports/uav_flow_official_10shard#' /home/ubuntu/vla/scripts/train_uav_flow_vla.py
grep -n "OFFICIAL_REPORT = " /home/ubuntu/vla/scripts/train_uav_flow_vla.py
R=/home/ubuntu/runs/official_k8_d169b
cd /home/ubuntu/vla
/opt/pytorch/bin/python scripts/train_uav_flow_vla.py --format official --chunk 8 --precision bf16 --instruction both --mirror \
  --batch-size 8 --accum 4 --workers 6 --updates 10000 --lr 5e-5 --schedule cosine \
  --val-every 500 --val-batches 1024 --sim-val-per-kind 56 --sim-val-examples 1024 --save-every 1000 --early-stop-patience 3 \
  --max-wall-seconds 68000 --init-adapter /home/ubuntu/runs/official_k8_realsim/adapter_s2000 \
  --wandb-project "vla training" --out $R
echo "TRAIN_EXIT=$?"
cd $R && tar -czf /home/ubuntu/d169_adapters.tgz $(ls -d adapter_* report.json sim_val_flights.json 2>/dev/null)
aws s3 cp /home/ubuntu/d169_adapters.tgz $BUCKET/d169_adapters.tgz
aws s3 cp $LOG $BUCKET/long_run_d169.log
echo PIPELINE_DONE
sudo shutdown -h now
