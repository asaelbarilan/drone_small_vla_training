"""Build a single-download D207 step8000 VLA package without changing weights."""
import argparse, hashlib, json, shutil
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",type=Path,required=True)
    ap.add_argument("--adapter",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(); root=Path(__file__).resolve().parents[2]
    if a.out.exists():raise SystemExit("Refusing to overwrite an existing package")
    expected="12e43d18489531d7446219bc88ed5ede0371935649b1464674a41e1cc155ac8e"
    if sha(a.adapter/"adapter_model.safetensors")!=expected:raise SystemExit("Not the verified step8000 adapter")
    a.out.mkdir(parents=True)
    shutil.copytree(a.base,a.out/"base",ignore=shutil.ignore_patterns(".cache",".git"))
    (a.out/"adapter").mkdir()
    for name in ["adapter_model.safetensors","adapter_config.json"]:shutil.copy2(a.adapter/name,a.out/"adapter"/name)
    # Preserve original metadata as provenance; make only the export portable.
    cfg=json.loads((a.out/"adapter/adapter_config.json").read_text())
    (a.out/"adapter_config_original.json").write_text(json.dumps(cfg,indent=2))
    cfg["base_model_name_or_path"]="Qwen/Qwen3-VL-4B-Instruct"
    (a.out/"adapter/adapter_config.json").write_text(json.dumps(cfg,indent=2))
    shutil.copytree(root/"scripts",a.out/"code/scripts",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    stats=a.out/"code/reports/uav_flow_official_10shard";stats.mkdir(parents=True)
    shutil.copy2(root/"reports/uav_flow_official_10shard/manifest.json",stats/"manifest.json")
    shutil.copy2(root/"requirements.txt",a.out/"requirements.txt")
    for name in ["serve.py","README.md"]:shutil.copy2(root/"release/d207_s8000_complete"/name,a.out/name)
    files=[{"path":p.relative_to(a.out).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(a.out.rglob("*")) if p.is_file()]
    manifest={"model":"D207 stage2 step8000 (D208 evaluation)","base":"Qwen/Qwen3-VL-4B-Instruct","base_revision":a.base.name,"layout":"complete base plus adapter, unmerged", "files":files,"full_gpu_inference_validation":"pending"}
    (a.out/"package_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"out":str(a.out),"files":len(files),"bytes":sum(x["bytes"] for x in files)}),flush=True)
if __name__=="__main__":main()
