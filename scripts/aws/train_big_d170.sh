#!/bin/bash
: "${VLA_BUCKET:?set VLA_BUCKET to your S3 bucket (see .env.example)}"
# D170 big run on a multi-GPU machine (planned: p4d.24xlarge, 8x A100 40 GB,
# DLAMI PyTorch, instance profile vla-eval-windows-role, SSM only). Needs the
# outputs of prep_more_d170.sh in s3://.../d170/ and the D169 adapters.
#
# 1. environment + data from S3 onto the local NVMe disk
# 2. 20-update multi-GPU smoke test (the DDP path has never run); abort if it fails
# 3. measure seconds per update, size the run to HOURS of training
# 4. one run, one warmup, cosine decay over the whole run, recipe lr 5e-4
#    (research note VLA_PLATEAU_20260929.md), validation every 1,000 updates with
#    loss and token accuracy, no early stop on loss
# 5. adapters synced to S3 every 30 min and at the end; machine stops itself.
# wandb runs offline (no credentials are copied to this machine); the wandb
# folder is uploaded for a later `wandb sync`.
set -x
HOURS="${HOURS:-5}"
LOG=/home/ubuntu/train_big_d170.log
exec >> $LOG 2>&1
sudo shutdown -h +$(( (HOURS + 2) * 60 )) "D170 big-run hard auto-stop"
BUCKET=s3://${VLA_BUCKET}/d170
NV=/opt/dlami/nvme
R=$NV/runs/big_d170
PY=/opt/pytorch/bin/python
GPUS=$(nvidia-smi -L | wc -l)
finish() {
  [ -d $R ] && aws s3 sync $R $BUCKET/run/ --exclude "checkpoint/*" --quiet
  aws s3 cp $LOG $BUCKET/train_big_d170.log
  sudo shutdown -h now; exit $1
}

# 1. Environment and data.
mkdir -p $NV/store $NV/init $NV/runs
aws s3 cp $BUCKET/vla_code.tgz - | tar -xzf - -C /home/ubuntu || finish 1
aws s3 cp $BUCKET/requirements_g5.txt /home/ubuntu/requirements_g5.txt
$PY -m pip uninstall -y transformer_engine transformer-engine-torch 2>/dev/null
$PY -m pip install -q $(grep -iE '^(transformers|peft|accelerate|safetensors|huggingface.hub|wandb)==' /home/ubuntu/requirements_g5.txt) || finish 1
$PY -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-VL-4B-Instruct', local_dir='$NV/qwen3vl4b')" || finish 1
aws s3 cp $BUCKET/store.tar - | tar -xf - -C $NV/store || finish 1
aws s3 cp s3://${VLA_BUCKET}/d169/d169_adapters.tgz - | tar -xzf - -C $NV/init adapter_best || finish 1
df -h $NV; nvidia-smi -L
export QWEN_MODEL=$NV/qwen3vl4b UAV_FLOW_OFFICIAL_STORE=$NV/store PYTHONPATH=/home/ubuntu/vla/src:/home/ubuntu/vla/scripts PYTHONWARNINGS=ignore WANDB_MODE=offline
cd /home/ubuntu/vla
COMMON="--format official --chunk 8 --precision bf16 --instruction both --mirror --batch-size 8 --accum 1 --workers 6 \
  --sim-val-per-kind 56 --init-adapter $NV/init/adapter_best --wandb-project vla_training"

# 2. Multi-GPU smoke test.
$PY -m torch.distributed.run --standalone --nproc_per_node $GPUS scripts/train_uav_flow_vla.py $COMMON \
  --updates 20 --lr 5e-4 --val-every 10 --val-batches 64 --sim-val-examples 64 --max-wall-seconds 3600 \
  --out $NV/runs/smoke || { echo SMOKE_FAILED; finish 1; }

# 3. Size the run from the measured speed (10% kept for validation passes).
SPU=$($PY -c "import json,statistics as s; r=json.load(open('$NV/runs/smoke/report.json')); print(s.median(r['seconds_per_update'][3:]))")
UPDATES=$($PY -c "print(int($HOURS * 3600 * 0.9 / $SPU))")
echo "seconds_per_update=$SPU updates=$UPDATES gpus=$GPUS examples_per_update=$((GPUS * 8))"

# 4. The run, with a background sync of adapters to S3.
( while sleep 1800; do aws s3 sync $R $BUCKET/run/ --exclude "checkpoint/*" --quiet; done ) &
$PY -m torch.distributed.run --standalone --nproc_per_node $GPUS scripts/train_uav_flow_vla.py $COMMON \
  --updates $UPDATES --lr 5e-4 --schedule cosine --warmup 0.03 --val-every 1000 --val-batches 1024 \
  --sim-val-examples 1024 --save-every 2000 --max-wall-seconds $(( HOURS * 3600 + 1800 )) --out $R
echo "TRAIN_EXIT=$?"
echo BIG_RUN_DONE
finish 0
