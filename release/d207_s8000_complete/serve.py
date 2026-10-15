"""Launch the bundled VLA server with local weights and the D208 inference settings."""
import argparse, os, subprocess, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--precision",choices=["bf16","nf4"],default="bf16")
    p.add_argument("--port",type=int,default=5007)
    p.add_argument("--log",type=Path,default=Path("outputs/server_calls.jsonl"))
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args();root=Path(__file__).resolve().parent
    for name in ["base/config.json","adapter/adapter_model.safetensors","code/reports/uav_flow_official_10shard/manifest.json"]:
        if not (root/name).is_file():raise SystemExit(f"Incomplete package: {name}")
    env=dict(os.environ,QWEN_MODEL=str(root/"base"),UAV_FLOW_OFFICIAL_REPORT=str(root/"code/reports/uav_flow_official_10shard"),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    cmd=[sys.executable,str(root/"code/scripts/uav_flow_eval_server.py"),"--model","qwen","--path",str(root/"adapter"),"--chunk","8","--precision",a.precision,"--port",str(a.port),"--log",str(a.log.resolve()),"--progress","--goal-memory","--memory-deadband","1.0,5","--box"]
    if a.dry_run:print(cmd);return
    raise SystemExit(subprocess.call(cmd,env=env))
if __name__=="__main__":main()
