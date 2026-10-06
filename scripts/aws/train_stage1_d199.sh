#!/bin/bash
: "${VLA_BUCKET:?set VLA_BUCKET to your S3 bucket (see .env.example)}"
# D199 = step 1 of the staged retrain (docs/PLAN.md D196, revised): boxes first.
# Fresh LoRA on the BASE Qwen3-VL-4B; answer = target box + box after the moves, no moves;
# the 81k D193 labels, balanced left / centre / right / empty (box_eval.balanced_sample);
# mirror copies and photo augmentation (BGR / 224 px, what the evaluator sends).
# Validation: written-box checks on 120 balanced held-out simulator rows every 250 updates
# (box/x_corr, box/box_centre_err vs box/centre_err_const in wandb). Gate for step 2: x_corr
# clearly above 0 and centre error below the always-centre baseline.
# Runs on the D193 machine (wandb login + packages kept on its disk); the NVMe data is re-fetched.
set -x
LOG=/home/ubuntu/train_stage1_d199.log
exec >> $LOG 2>&1
sudo shutdown -h +300 "D199 hard auto-stop"
S3=s3://${VLA_BUCKET}
B=$S3/d199
D=/opt/dlami/nvme; [ -d $D ] || D=/home/ubuntu
sudo chown ubuntu:ubuntu $D
PY=/opt/pytorch/bin/python
STORE=$D/uav_flow_official_10shard
R=$D/runs/stage1_d199
export QWEN_MODEL=$D/qwen3vl4b UAV_FLOW_OFFICIAL_STORE=$STORE PYTHONPATH=/home/ubuntu/vla/src:/home/ubuntu/vla/scripts PYTHONWARNINGS=ignore
step() { echo "=== $1 $(date -u)"; aws s3 cp $LOG $B/train_stage1_d199.log --quiet; }
finish() { aws s3 sync $R $B/stage1/ --exclude "checkpoint/*" --exclude "wandb/*" --quiet; aws s3 cp $LOG $B/train_stage1_d199.log; sudo shutdown -h now; exit; }

step "1 setup"
aws s3 cp $B/vla_code.zip /home/ubuntu/vla_code.zip && $PY -c "import zipfile; zipfile.ZipFile('/home/ubuntu/vla_code.zip').extractall('/home/ubuntu/vla')" || finish
$PY -c "import peft, transformers, wandb; print('peft', peft.__version__, 'transformers', transformers.__version__, 'wandb', wandb.__version__)" || finish
[ -f $QWEN_MODEL/config.json ] || $PY -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-VL-4B-Instruct', local_dir='$QWEN_MODEL')" || finish
mkdir -p $STORE
[ -f $STORE/episodes.jsonl ] || aws s3 cp $S3/d170/store.tar - | tar -x -C $STORE || finish
wc -l $STORE/episodes.jsonl || finish
aws s3 cp $S3/d193/boxes.jsonl $D/boxes.jsonl --quiet || finish
wc -l $D/boxes.jsonl

step "2 stage 1 training (boxes only)"
cd /home/ubuntu/vla
( while sleep 1800; do aws s3 sync $R $B/stage1/ --exclude "checkpoint/*" --exclude "wandb/*" --quiet; aws s3 cp $LOG $B/train_stage1_d199.log --quiet; done ) &
$PY scripts/train_uav_flow_vla.py --format official --chunk 8 --precision bf16 --instruction both --mirror --progress \
  --boxes $D/boxes.jsonl --next-box --grounding-only --photo-aug \
  --batch-size 8 --accum 4 --workers 6 --updates 1000 --lr 2e-4 --schedule cosine --warmup 0.03 \
  --sim-val-per-kind 56 --sim-val-examples 64 --val-every 250 --val-batches 32 --box-val 120 \
  --save-every 250 --max-wall-seconds 14400 --wandb-project "vla training" --out $R
echo "TRAIN_EXIT=$?"
step "3 done"
echo PIPELINE_DONE
finish
