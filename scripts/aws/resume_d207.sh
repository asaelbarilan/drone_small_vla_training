#!/usr/bin/env bash
# D207: user-approved continuation of D204 checkpoint 5000 to TOTAL update 8000.
# Keep all data/mix/schedule settings unchanged. Outputs isolated from D204.
set -Eeuo pipefail
: "${VLA_BUCKET:?set VLA_BUCKET}"
LOG=/home/ubuntu/resume_d207.log
exec >> "$LOG" 2>&1
D=/opt/dlami/nvme
[ -d "$D" ] || D=/home/ubuntu
PY=/opt/pytorch/bin/python
CODE=/home/ubuntu/vla_d207
STORE=$D/uav_flow_official_10shard
R=$D/runs/stage2_d207
B=s3://${VLA_BUCKET}/d207
S3=s3://${VLA_BUCKET}
SYNC_PID=""
finish() {
  rc=$?
  trap - EXIT INT TERM
  set +e
  if [ -n "$SYNC_PID" ]; then kill "$SYNC_PID" 2>/dev/null; fi
  echo "D207_EXIT=$rc $(date -u)"
  # Bound final uploads; shutdown still happens on upload failure.
  timeout 900 aws s3 sync "$R" "$B/stage2/" --exclude 'wandb/*' --quiet
  upload_rc=$?
  echo "FINAL_UPLOAD_EXIT=$upload_rc $(date -u)"
  timeout 120 aws s3 cp "$LOG" "$B/resume_d207.log" --quiet
  sudo shutdown -h now
  exit "$rc"
}
# Set deadline before any lengthy setup; never cancel it while training runs.
sudo shutdown -h +720 "D207 hard stop: 12 hours maximum"
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
sudo chown ubuntu:ubuntu "$D"
mkdir -p "$CODE" "$STORE" "$R/checkpoint"
echo "D207_SETUP $(date -u)"
aws s3 cp "$S3/d204/vla_code.zip" /home/ubuntu/vla_code_d207.zip --quiet
"$PY" -c "import zipfile; zipfile.ZipFile('/home/ubuntu/vla_code_d207.zip').extractall('$CODE')"
export QWEN_MODEL=$D/qwen3vl4b UAV_FLOW_OFFICIAL_STORE=$STORE
export PYTHONPATH=$CODE/src:$CODE/scripts PYTHONWARNINGS=ignore
"$PY" -c "import torch, peft, transformers, wandb; assert torch.cuda.is_available(); print('GPU',torch.cuda.get_device_name(0))"
if [ ! -f "$QWEN_MODEL/config.json" ]; then
  "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-4B-Instruct',local_dir='$QWEN_MODEL')"
fi
if [ ! -f "$STORE/episodes.jsonl" ]; then
  aws s3 cp "$S3/d170/store.tar" - | tar -xf - -C "$STORE"
fi
test -s "$STORE/episodes.jsonl"
aws s3 cp "$S3/d193/boxes.jsonl" "$D/boxes.jsonl" --quiet
aws s3 sync "$S3/d204/stage2/checkpoint/" "$R/checkpoint/" --quiet
aws s3 cp "$S3/d204/stage2/report.json" "$R/source_report.json" --quiet
"$PY" - "$R" <<'PY'
import sys,json,torch
from pathlib import Path
r=Path(sys.argv[1]); s=torch.load(r/'checkpoint/state.pt',map_location='cpu',weights_only=False)
assert s['step']==5000, f"Expected checkpoint 5000, got {s['step']}"
assert 'optimiser' in s and 'report' in s
(r/'report.json').write_text(json.dumps(s['report']))
(r/'continuation.json').write_text(json.dumps(dict(decision='D207',source='d204/stage2/checkpoint',start_step=5000,total_updates=8000,grounding_mix=0.1,sim_share=0.5,vision_lr_mult=0.1,schedule='cosine',lr=2e-4,warmup=0.03)))
print('CHECKPOINT_VERIFIED step=5000 optimizer restored by --resume',flush=True)
PY
cd "$CODE"
ROWS=()
for T in d188_dagger_r1 d190_dagger_r1; do
  aws s3 cp "$S3/d172/$T.zip" "$D/$T.zip" --quiet
  "$PY" - "$D/$T.zip" "$D/$T" <<'PY'
import sys,zipfile
from pathlib import Path
out=Path(sys.argv[2]).resolve()
with zipfile.ZipFile(sys.argv[1]) as z:
    for info in z.infolist():
        name=info.filename.replace(chr(92),'/')
        if name.endswith('/'): continue
        p=(out/name).resolve()
        if not p.is_relative_to(out): raise ValueError('Archive path outside destination')
        p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(z.read(info))
PY
  "$PY" scripts/rl/dagger_relabel.py --flights "$D/$T/pass1" --calls "$D/$T/server_calls.jsonl" --images "$D/$T/rollout_images" --tasks "$D/$T/tasks" --out "$D/$T.rows.jsonl"
  test -s "$D/$T.rows.jsonl"
  ROWS+=("$D/$T.rows.jsonl")
done
COMMON=(--format official --chunk 8 --precision bf16 --instruction both --mirror --progress
 --boxes "$D/boxes.jsonl" --box-repeat 3 --next-box --photo-aug
 --extra-rows "${ROWS[@]}" --extra-repeat 20
 --grounding-mix 0.1 --sim-share 0.5 --vision-lr-mult 0.1
 --init-adapter "$R/checkpoint" --batch-size 8 --accum 4 --workers 6
 --sim-val-per-kind 56 --sim-val-examples 1024 --box-val 120)
# Mid-run uploads omit the mutable checkpoint; final upload includes it after trainer exit.
( trap - EXIT INT TERM; while sleep 1800; do
    aws s3 sync "$R" "$B/stage2/" --exclude 'wandb/*' --exclude 'checkpoint/*' --quiet || true
    aws s3 cp "$LOG" "$B/resume_d207.log" --quiet || true
  done ) &
SYNC_PID=$!
echo "D207_TRAIN_START checkpoint=5000 target=8000 $(date -u)"
aws s3 cp "$LOG" "$B/resume_d207.log" --quiet
"$PY" scripts/train_uav_flow_vla.py "${COMMON[@]}" --resume --updates 8000 --lr 2e-4 --schedule cosine --warmup 0.03 --val-every 1000 --val-batches 512 --save-every 500 --max-wall-seconds 36000 --wandb-project "vla training" --out "$R"
"$PY" - "$R" <<'PY'
import sys,json,torch
from pathlib import Path
r=Path(sys.argv[1]); s=torch.load(r/'checkpoint/state.pt',map_location='cpu',weights_only=False)
report=json.loads((r/'report.json').read_text())
assert s['step']==8000, f"Training exited before 8000: checkpoint={s['step']}"
assert report['status']=='complete'
assert (r/'adapter_s8000/adapter_model.safetensors').exists()
print('D207_COMPLETE step=8000',flush=True)
PY
