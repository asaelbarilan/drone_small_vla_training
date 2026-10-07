#!/bin/bash
: "${VLA_BUCKET:?set VLA_BUCKET to your S3 bucket (see .env.example)}"
# D204 = step 2 of the staged retrain AGAIN, with the 8 moves as bin tokens (as in every model
# that flew, D190 69 %) instead of the D201 action head: the head could not output "still" /
# "no turn" and drifted worse with training (D203: 12 % on validation, never stopped).
# Continues the step-1 LoRA (d199/stage1/adapter_s1000); full answer = box + box after the moves
# (labelled photos only) + Left line + 32 move tokens. Everything else exactly as D202.
# Keeps the D193 checklist: --progress, --mirror, --photo-aug, --boxes (x3), DAgger rows of
# rounds 1 and 2 (rebuilt here from their rollout zips), both wordings, 504 held-out sim flights.
# New from D196-D198: 1 box-only example per 10 flight examples (ST4VLA), simulator = half of the
# flight examples, vision LoRA at 0.1x learning rate, written-box checks at every validation.
# 20-update smoke first, then 8,000 updates; validation + adapter every 1,000 (the step-2 gate:
# move accuracy rising AND box/* not falling - checked by hand). Machine off at the end.
# Runs on the D193/D199 machine (wandb login + packages on its root disk); NVMe data re-fetched.
set -x
LOG=/home/ubuntu/train_stage2_d204.log
exec >> $LOG 2>&1
sudo shutdown -h +1680 "D204 hard auto-stop"
S3=s3://${VLA_BUCKET}
B=$S3/d204
D=/opt/dlami/nvme; [ -d $D ] || D=/home/ubuntu
sudo chown ubuntu:ubuntu $D
PY=/opt/pytorch/bin/python
STORE=$D/uav_flow_official_10shard
R=$D/runs/stage2_d204
export QWEN_MODEL=$D/qwen3vl4b UAV_FLOW_OFFICIAL_STORE=$STORE PYTHONPATH=/home/ubuntu/vla/src:/home/ubuntu/vla/scripts PYTHONWARNINGS=ignore
step() { echo "=== $1 $(date -u)"; aws s3 cp $LOG $B/train_stage2_d204.log --quiet; }
finish() { aws s3 sync $R $B/stage2/ --exclude "wandb/*" --quiet; aws s3 cp $LOG $B/train_stage2_d204.log; sudo shutdown -h now; exit; }

step "1 setup"
rm -rf /home/ubuntu/vla
aws s3 cp $B/vla_code.zip /home/ubuntu/vla_code.zip && $PY -c "import zipfile; zipfile.ZipFile('/home/ubuntu/vla_code.zip').extractall('/home/ubuntu/vla')" || finish
$PY -c "import peft, transformers, wandb; print('peft', peft.__version__, 'transformers', transformers.__version__, 'wandb', wandb.__version__)" || finish
[ -f $QWEN_MODEL/config.json ] || $PY -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-VL-4B-Instruct', local_dir='$QWEN_MODEL')" || finish
mkdir -p $STORE
[ -f $STORE/episodes.jsonl ] || aws s3 cp $S3/d170/store.tar - | tar -x -C $STORE || finish
wc -l $STORE/episodes.jsonl || finish
aws s3 cp $S3/d193/boxes.jsonl $D/boxes.jsonl --quiet || finish
aws s3 sync $S3/d199/stage1/adapter_s1000/ $D/start_adapter/ --quiet
[ -f $D/start_adapter/adapter_config.json ] || finish
df -h $D | tail -1

step "2 DAgger rows (rounds 1 and 2)"
cd /home/ubuntu/vla
ROWS=""
for T in d188_dagger_r1 d190_dagger_r1; do
  aws s3 cp $S3/d172/$T.zip $D/$T.zip --quiet && $PY - "$D/$T.zip" "$D/$T" <<'EOF'
import sys, zipfile, pathlib
src, out = sys.argv[1], pathlib.Path(sys.argv[2])
with zipfile.ZipFile(src) as z:
    for i in z.infolist():
        name = i.filename.replace("\\", "/")
        if name.endswith("/"):
            continue
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(z.read(i))
EOF
  $PY scripts/rl/dagger_relabel.py --flights $D/$T/pass1 --calls $D/$T/server_calls.jsonl \
      --images $D/$T/rollout_images --tasks $D/$T/tasks --out $D/$T.rows.jsonl && ROWS="$ROWS $D/$T.rows.jsonl"
done
echo "DAgger rows: $ROWS"; wc -l $ROWS
[ -n "$ROWS" ] || finish

COMMON="--format official --chunk 8 --precision bf16 --instruction both --mirror --progress \
  --boxes $D/boxes.jsonl --box-repeat 3 --next-box --photo-aug \
  --extra-rows $ROWS --extra-repeat 20 \
  --grounding-mix 0.1 --sim-share 0.5 --vision-lr-mult 0.1 \
  --init-adapter $D/start_adapter --batch-size 8 --accum 4 --workers 6 \
  --sim-val-per-kind 56 --sim-val-examples 1024 --box-val 120"

step "3 smoke (20 updates)"
$PY scripts/train_uav_flow_vla.py $COMMON --updates 20 --lr 2e-4 --schedule constant \
  --val-every 20 --val-batches 16 --sim-val-examples 64 --max-wall-seconds 3600 --out $D/runs/smoke_d204 || finish

step "4 stage 2 training (8,000 updates, lr 2e-4, warmup 3 %, cosine to 0)"
( while sleep 1800; do aws s3 sync $R $B/stage2/ --exclude "wandb/*" --quiet; aws s3 cp $LOG $B/train_stage2_d204.log --quiet; done ) &
$PY scripts/train_uav_flow_vla.py $COMMON --updates 8000 --lr 2e-4 --schedule cosine --warmup 0.03 \
  --val-every 1000 --val-batches 512 --save-every 1000 --max-wall-seconds 90000 \
  --wandb-project "vla training" --out $R
echo "TRAIN_EXIT=$?"
step "5 done"
echo PIPELINE_DONE
finish
