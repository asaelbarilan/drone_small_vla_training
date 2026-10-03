#!/bin/bash
: "${VLA_BUCKET:?set VLA_BUCKET to your S3 bucket (see .env.example)}"
# D175 progress-token fine-tune on the Linux g5, from a D171 adapter (INIT).
#   1. new code from S3 (d175/vla_code.zip: scripts, src, 10-shard manifest);
#   2. expert-flight store for the Windows RL loop (export_anchor_store.py), to S3;
#   3. smoke: 30 updates with --progress, then the server end to end (greedy +
#      goal memory, and sampling) - stops here if either fails;
#   4. the fine-tune: UPDATES updates, same data and validation as D171, with
#      --progress unless PROGRESS=0 (the matched "without" run for the paper);
#   5. adapters to S3, machine off. Hard stop after 20 h.
# Usage: INIT=/home/ubuntu/runs/official_k8_d171/adapter_s26000 PROGRESS=1 UPDATES=5000 bash train_progress_d175.sh
set -x
INIT=${INIT:?set INIT to the D171 adapter}
PROGRESS=${PROGRESS:-1}
UPDATES=${UPDATES:-5000}
NAME=progress_d175_p${PROGRESS}
LOG=/home/ubuntu/$NAME.log
exec >> $LOG 2>&1
sudo shutdown -h +1200 "D175 hard auto-stop"
BUCKET=s3://${VLA_BUCKET}/d175
R=/home/ubuntu/runs/$NAME
PY=/opt/pytorch/bin/python
export QWEN_MODEL=/home/ubuntu/qwen3vl4b UAV_FLOW_OFFICIAL_STORE=/home/ubuntu/data/uav_flow_official_10shard PYTHONPATH=/home/ubuntu/vla/src:/home/ubuntu/vla/scripts PYTHONWARNINGS=ignore
aws s3 cp $BUCKET/vla_code.zip /home/ubuntu/vla_code.zip && cd /home/ubuntu/vla && $PY -m zipfile -e /home/ubuntu/vla_code.zip .
grep -n "OFFICIAL_REPORT" scripts/train_uav_flow_vla.py | head -3
$PY -c "import flask" 2>/dev/null || $PY -m pip install -q flask

aws s3 cp s3://${VLA_BUCKET}/d172/rl_tasks.zip /home/ubuntu/rl_tasks.zip && \
  $PY scripts/rl/export_anchor_store.py --store $UAV_FLOW_OFFICIAL_STORE --tasks-zip /home/ubuntu/rl_tasks.zip \
    --out /home/ubuntu/anchor_store.tar && aws s3 cp /home/ubuntu/anchor_store.tar $BUCKET/anchor_store.tar

FLAG=""; [ "$PROGRESS" = "1" ] && FLAG="--progress"
COMMON="--format official --chunk 8 --precision bf16 --instruction both --mirror --batch-size 8 --accum 4 --workers 6 --sim-val-per-kind 56 --sim-val-examples 1024 $FLAG"
if [ "$PROGRESS" = "1" ]; then
  $PY scripts/train_uav_flow_vla.py $COMMON --updates 30 --lr 2e-4 --val-every 30 --val-batches 64 \
    --init-adapter $INIT --max-wall-seconds 3600 --out /home/ubuntu/runs/${NAME}_smoke || { echo SMOKE_TRAIN_FAILED; sudo shutdown -h now; exit 1; }
  SMOKE=$(ls -d /home/ubuntu/runs/${NAME}_smoke/adapter_s* | tail -1)
  $PY scripts/smoke_progress_server.py --adapter $SMOKE --goal-memory --out /home/ubuntu/runs/${NAME}_smoke/server_greedy \
    && $PY scripts/smoke_progress_server.py --adapter $SMOKE --temperature 1.0 --out /home/ubuntu/runs/${NAME}_smoke/server_sample \
    || { echo SMOKE_SERVER_FAILED; aws s3 cp $LOG $BUCKET/$NAME.log; sudo shutdown -h now; exit 1; }
  echo SMOKE_OK
fi

( while sleep 3600; do aws s3 sync $R $BUCKET/$NAME/ --exclude "checkpoint/*" --exclude "wandb/*" --quiet; done ) &
$PY scripts/train_uav_flow_vla.py $COMMON --updates $UPDATES --lr 2e-4 --schedule cosine --warmup 0.03 \
  --val-every 1000 --val-batches 1024 --save-every 1000 --max-wall-seconds 64800 \
  --init-adapter $INIT --wandb-project "vla training" --out $R
echo "TRAIN_EXIT=$?"
aws s3 sync $R $BUCKET/$NAME/ --exclude "checkpoint/*" --exclude "wandb/*" --quiet
aws s3 cp $LOG $BUCKET/$NAME.log
echo PIPELINE_DONE
sudo shutdown -h now
