#!/bin/bash
: "${VLA_BUCKET:?set VLA_BUCKET to your S3 bucket (see .env.example)}"
# D165 pipeline on the Linux g5: sim data -> prepare -> continue training -> S3 -> stop.
set -x
LOG=/home/ubuntu/sim_pipeline.log
exec >> $LOG 2>&1
sudo shutdown -h +360 "hard auto-stop"
rm -rf /home/ubuntu/sim/Collection_v4_LinuxNoEditor /home/ubuntu/sim/runs /home/ubuntu/sim/openvla-uav
df -h /
export QWEN_MODEL=/home/ubuntu/qwen3vl4b UAV_FLOW_OFFICIAL_STORE=/home/ubuntu/data/uav_flow_official_10shard PYTHONPATH=/home/ubuntu/vla/src:/home/ubuntu/vla/scripts PYTHONWARNINGS=ignore
/opt/pytorch/bin/python -c "
from huggingface_hub import snapshot_download as d
d('wangxiangyu0814/UAV-Flow-Sim', repo_type='dataset', local_dir='/home/ubuntu/data/uav_flow_sim_raw', max_workers=16)
print('SIM_DOWNLOADED')"
cd /home/ubuntu/vla
/opt/pytorch/bin/python scripts/prepare_uav_flow_sim.py --shard /home/ubuntu/data/uav_flow_sim_raw/*.parquet --exclude /home/ubuntu/uav_flow_sim_excluded.json --manifest /home/ubuntu/vla/reports/uav_flow_official_10shard/manifest.json --workers 8 || { echo PREP_FAILED; aws s3 cp $LOG s3://${VLA_BUCKET}/d165/sim_pipeline.log; sudo shutdown -h now; exit 1; }
rm -rf /home/ubuntu/data/uav_flow_sim_raw
/opt/pytorch/bin/python scripts/train_uav_flow_vla.py --format official --chunk 8 --precision bf16 --instruction both --mirror \
  --batch-size 8 --accum 4 --workers 6 --updates 2000 --lr 2e-4 --schedule cosine --val-every 250 --val-batches 128 --save-every 500 \
  --max-wall-seconds 16000 --init-adapter /home/ubuntu/runs/official_k8_10shard/adapter_s2500 \
  --wandb-project "vla training" --out /home/ubuntu/runs/official_k8_realsim
R=/home/ubuntu/runs/official_k8_realsim
LAST=$(ls -d $R/adapter_s* | sort -V | tail -1)
cd $R && tar -czf /home/ubuntu/realsim_adapter.tgz $(basename $LAST) report.json /home/ubuntu/data/uav_flow_official_10shard/sim_added.json
aws s3 cp /home/ubuntu/realsim_adapter.tgz s3://${VLA_BUCKET}/d165/realsim_adapter.tgz
aws s3 cp $LOG s3://${VLA_BUCKET}/d165/sim_pipeline.log
echo PIPELINE_DONE
sudo shutdown -h now
