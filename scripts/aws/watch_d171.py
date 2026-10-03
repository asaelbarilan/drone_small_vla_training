"""D171 guard: stop the long run if the simulator validation clearly gets worse.

Reads the trainer's report.json every 5 minutes. If the simulator validation
action-token accuracy is more than 2 points below its best for 3 validations in
a row (3,000 updates, about 5 hours), it stops the training process; the run
script then uploads the adapters and the log and shuts the machine down. It does
not stop anything when the numbers are merely flat. Exits when training ends.
"""

import json
import subprocess
import time
from pathlib import Path

REPORT = Path("/home/ubuntu/runs/official_k8_d171/report.json")
LOG = Path("/home/ubuntu/watch_d171.log")
MARGIN, PATIENCE = 0.02, 3


def log(message):
    with LOG.open("a") as f:
        f.write(time.strftime("%H:%M ") + message + "\n")


def training_alive():
    return subprocess.run(["pgrep", "-f", "[t]rain_uav_flow_vla.py"], capture_output=True).returncode == 0


best, bad, seen = -1.0, 0, set()
log("watcher started")
while True:
    time.sleep(300)
    if not training_alive():
        log("training ended; watcher exits")
        break
    try:
        entries = json.loads(REPORT.read_text())["held_out"]
    except (OSError, ValueError, KeyError):
        continue
    for entry in entries:
        if entry["step"] in seen or "sim_token_acc" not in entry:
            continue
        seen.add(entry["step"])
        accuracy = entry["sim_token_acc"]
        if accuracy > best:
            best, bad = accuracy, 0
        elif accuracy < best - MARGIN:
            bad += 1
        else:
            bad = 0
        log(f"step {entry['step']}: sim accuracy {accuracy:.4f}, best {best:.4f}, checks below {bad}")
        if bad >= PATIENCE:
            log("STOPPING: accuracy stayed more than 2 points below its best for 3 checks")
            subprocess.run(["pkill", "-f", "[t]rain_uav_flow_vla.py"])
            raise SystemExit
