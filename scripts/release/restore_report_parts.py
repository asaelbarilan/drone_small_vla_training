"""Reassemble downloaded report parts, validating each part and the whole archive."""
import argparse,hashlib,json,os
from pathlib import Path,PurePosixPath

def restore(manifest_path,parts_dir,output):
    m=json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    output=Path(output);temp=output.with_name(output.name+".partial")
    if output.exists() or temp.exists():raise FileExistsError("Output or partial file already exists; refusing overwrite")
    whole=hashlib.sha256();total=0
    try:
        with temp.open("xb") as out:
            for number,p in enumerate(m["parts"],1):
                if p["part"]!=number or p["offset"]!=total:raise ValueError("Noncontiguous part manifest")
                h=hashlib.sha256();size=0
                source=Path(parts_dir)/PurePosixPath(p["key"]).name
                with source.open("rb") as f:
                    for b in iter(lambda:f.read(8*1024*1024),b""):
                        h.update(b);whole.update(b);out.write(b);size+=len(b)
                if size!=p["bytes"] or h.hexdigest()!=p["sha256"]:raise ValueError(f"Part {number} checksum/size mismatch")
                total+=size
            if total!=m["bytes"] or whole.hexdigest()!=m["sha256"]:raise ValueError("Archive checksum/size mismatch")
        os.rename(temp,output)
    except Exception:
        if temp.exists():temp.unlink()
        raise
    return {"bytes":total,"sha256":whole.hexdigest()}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("manifest",type=Path);p.add_argument("parts",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();print(json.dumps(restore(a.manifest,a.parts,a.output)))
