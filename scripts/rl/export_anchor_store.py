"""D175 RL: a small store holding only the practice tasks' own recorded flights.

The GRPO update adds an SFT anchor on each task's source flight (EG-GRPO /
WorldVLN's 12:1 mix), which needs those flights' frames. The full store is 75 GB
and lives on the Linux g5; the RL loop runs on the Windows simulator box (only one
G machine fits the GPU quota at a time). This copies the source flights of every
practice task (episodes.jsonl rows + frames) into one tar that the loop unpacks as
its UAV_FLOW_OFFICIAL_STORE. Run on the g5, where the full store is.
"""

import argparse
import io
import json
import tarfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True, help="full official store")
    parser.add_argument("--tasks-zip", type=Path, required=True, help="rl_tasks.zip")
    parser.add_argument("--out", type=Path, required=True, help="tar to write")
    args = parser.parse_args()

    with zipfile.ZipFile(args.tasks_zip) as archive:
        sources = {
            json.loads(archive.read(name))["source_flight"]
            for name in archive.namelist()
            if name.endswith(".json")
        }
    kept, frames = [], 0
    with tarfile.open(args.out, "w") as tar:
        for line in (args.store / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            episode = json.loads(line)
            if episode["episode"] not in sources:
                continue
            kept.append(line)
            for image in episode["images"]:
                path = Path(image.replace("\\", "/"))
                local = args.store / "frames" / path.parent.name / path.name
                tar.add(local, arcname=f"frames/{path.parent.name}/{path.name}")
                frames += 1
        data = ("\n".join(kept) + "\n").encode("utf-8")
        info = tarfile.TarInfo("episodes.jsonl")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    missing = sorted(sources - {json.loads(line)["episode"] for line in kept})
    print(json.dumps(dict(tasks_sources=len(sources), episodes=len(kept), frames=frames,
                          missing=missing[:5], missing_count=len(missing))))


if __name__ == "__main__":
    main()
