#!/bin/bash
: "${VLA_BUCKET:?set VLA_BUCKET to your S3 bucket (see .env.example)}"
# D170 data prep on the Linux g5 (cheap machine): add the 44 unused real shards
# (train-00010..00053 of 54) to the store, then stream the whole store to S3 for
# the big training machine. Stops the machine when done or on failure; hard stop
# after 12 h. Run as ubuntu (wandb credentials live in ~/.netrc).
#
# Before starting: the root volume must already be enlarged (EBS modify-volume,
# done from the AWS API); this script only grows the partition and filesystem.
# SWEEP="--instruction-sweep" keeps the old rule that drops train flights whose
# wording also appears in validation (~44% of flights); empty keeps them.
set -x
SWEEP="${SWEEP:-}"
LOG=/home/ubuntu/prep_more_d170.log
exec >> $LOG 2>&1
sudo shutdown -h +720 "D170 prep hard auto-stop"
BUCKET=s3://${VLA_BUCKET}/d170
STORE=/home/ubuntu/data/uav_flow_official_10shard
export QWEN_MODEL=/home/ubuntu/qwen3vl4b UAV_FLOW_OFFICIAL_STORE=$STORE PYTHONPATH=/home/ubuntu/vla/src:/home/ubuntu/vla/scripts PYTHONWARNINGS=ignore
PY=/opt/pytorch/bin/python
finish() { aws s3 cp $LOG $BUCKET/prep_more_d170.log; sudo shutdown -h now; exit $1; }

# 1. Grow the root partition and filesystem to the enlarged volume.
ROOT_PART=$(findmnt -n -o SOURCE /)
ROOT_DISK=/dev/$(lsblk -no PKNAME "$ROOT_PART")
PART_NO=$(cat /sys/class/block/$(basename "$ROOT_PART")/partition)
sudo growpart "$ROOT_DISK" "$PART_NO"; sudo resize2fs "$ROOT_PART"
df -h / /opt/dlami/nvme

# 2. Code from S3 (the trainer carries the 10-shard manifest path edit).
for f in train_uav_flow_vla.py prepare_uav_flow_more.py; do aws s3 cp $BUCKET/$f /home/ubuntu/vla/scripts/$f; done
sed -i 's#reports/uav_flow_official_20260922#reports/uav_flow_official_10shard#' /home/ubuntu/vla/scripts/train_uav_flow_vla.py
cd /home/ubuntu/vla
$PY -m pip freeze > /home/ubuntu/requirements_g5.txt && aws s3 cp /home/ubuntu/requirements_g5.txt $BUCKET/requirements_g5.txt

# 3. Smoke test of the trainer (validation, accuracy, simulator split): 3 updates.
$PY scripts/train_uav_flow_vla.py --format official --chunk 8 --precision bf16 --instruction both --mirror \
  --batch-size 8 --accum 1 --workers 4 --updates 3 --lr 5e-4 --val-every 2 --val-batches 16 \
  --sim-val-per-kind 56 --sim-val-examples 32 --max-wall-seconds 1800 \
  --init-adapter /home/ubuntu/runs/official_k8_realsim/adapter_s2000 --out /home/ubuntu/runs/smoke_d170 \
  || { echo SMOKE_FAILED; finish 1; }
cat /home/ubuntu/runs/smoke_d170/report.json | $PY -c "import json,sys; r=json.load(sys.stdin); print(r['held_out'])"
rm -rf /home/ubuntu/runs/smoke_d170

# 4. The 44 unused real shards, one at a time through the NVMe scratch disk.
RAW=/opt/dlami/nvme/raw && mkdir -p $RAW
for i in $(seq 10 53); do
  NAME=$(printf "train-%05d-of-00054.parquet" $i)
  $PY -c "from huggingface_hub import hf_hub_download as d; d('wangxiangyu0814/UAV-Flow', '$NAME', repo_type='dataset', local_dir='$RAW')" \
    || { echo DOWNLOAD_FAILED $NAME; finish 1; }
  $PY scripts/prepare_uav_flow_more.py --shard "$RAW/$NAME" --manifest reports/uav_flow_official_10shard/manifest.json \
    --workers 8 --delete-shards $SWEEP || { echo PREP_FAILED $NAME; finish 1; }
  df -h / | tail -1
done
cp $STORE/more_added.json /home/ubuntu/more_added.json && aws s3 cp /home/ubuntu/more_added.json $BUCKET/more_added.json

# 5. Stream the store to S3 as one tar (no temp copy needed) plus the code.
cd $STORE && tar -cf - . | aws s3 cp - $BUCKET/store.tar --expected-size 200000000000 || { echo UPLOAD_FAILED; finish 1; }
cd /home/ubuntu && tar -czf vla_code.tgz vla/scripts vla/src vla/reports/uav_flow_official_10shard && aws s3 cp vla_code.tgz $BUCKET/vla_code.tgz
echo PREP_DONE
finish 0
