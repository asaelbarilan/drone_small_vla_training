#!/bin/bash
: "${VLA_BUCKET:?set VLA_BUCKET to your S3 bucket (see .env.example)}"
# D193 restart of the training step only, now logged to Weights & Biases (user, 2026-10-05).
# The first training start (13:23 UTC) ran without wandb and was stopped at update 210 on
# request. Everything else from train_box_d193.sh is reused from this machine's NVMe disk:
# store, base model, round-2 start adapter, DAgger rows, box labels. Needs `wandb login`
# done once as ubuntu (by the user). Same flags as before, plus --wandb-project.
set -x
LOG=/home/ubuntu/train_only_d193.log
exec >> $LOG 2>&1
S3=s3://${VLA_BUCKET}
B=$S3/d193
D=/opt/dlami/nvme
PY=/opt/pytorch/bin/python
R=$D/runs/box_d193
export QWEN_MODEL=$D/qwen3vl4b UAV_FLOW_OFFICIAL_STORE=$D/uav_flow_official_10shard PYTHONPATH=/home/ubuntu/vla/src:/home/ubuntu/vla/scripts PYTHONWARNINGS=ignore
finish() { aws s3 sync $R $B/run/ --exclude "checkpoint/*" --exclude "wandb/*" --quiet; aws s3 cp $LOG $B/train_only_d193.log; sudo shutdown -h now; exit; }
$PY -c "import wandb; print('wandb', wandb.__version__)" || { echo "wandb missing"; exit 1; }
rm -rf $R
cd /home/ubuntu/vla
( while sleep 3600; do aws s3 sync $R $B/run/ --exclude "checkpoint/*" --exclude "wandb/*" --quiet; aws s3 cp $LOG $B/train_only_d193.log --quiet; done ) &
$PY scripts/train_uav_flow_vla.py --format official --chunk 8 --precision bf16 --instruction both --mirror --progress \
  --boxes $D/boxes.jsonl --box-repeat 3 --photo-aug --next-box \
  --extra-rows $D/d188_dagger_r1.rows.jsonl $D/d190_dagger_r1.rows.jsonl --extra-repeat 20 \
  --init-adapter $D/start_adapter --batch-size 8 --accum 4 --workers 6 \
  --sim-val-per-kind 56 --sim-val-examples 1024 \
  --updates 4000 --lr 1e-4 --schedule cosine --warmup 0.03 \
  --val-every 500 --val-batches 1024 --save-every 1000 --max-wall-seconds 57600 \
  --wandb-project "vla training" --out $R
echo "TRAIN_EXIT=$?"
echo PIPELINE_DONE
finish
