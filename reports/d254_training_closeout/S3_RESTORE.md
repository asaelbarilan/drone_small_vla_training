# Restore inventory (D255)

Use the configured private bucket. No machine disks are required for these files once all pending uploads are verified.

## Complete VLA inference package

1. Extract `d255/full_vla/package_assets.zip` into an empty directory.
2. Download `d255/full_vla/base/model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors` into its `base/` directory.
3. Download `d255/full_vla/adapter/adapter_model.safetensors` into its `adapter/` directory.
4. Verify every file against `package_manifest.json`. Install `requirements.txt`, then `python serve.py --precision bf16` on a compatible CUDA machine.

The base and adapter are included together, unmerged. Full GPU release smoke and license/notices review remain pending; offline processor/decoder/launch checks passed. This package is not yet on Hugging Face.

## Training restart

Retain original `d207/stage2/checkpoint/` (adapter/config/state.pt), `d207/stage2/report.json`, `d207/stage2/sim_val_flights.json`, `d204/vla_code.zip`, `d170/store.tar`, `d193/boxes.jsonl`, and `d172/d188_dagger_r1.zip` plus `d172/d190_dagger_r1.zip`. Exact D207 launch arguments are in `scripts/aws/resume_d207.sh`; review/update paths and budgets before executing anything. Do not blindly rerun its historical cloud launch. The dataset was not freshly re-downloaded and a full restore was not exercised in this closeout.

## Research and source

GitHub main contains source/tests/research. `d255/closeout_supplement.zip` includes Git bundle and latest closeout documents. Full historical reports/source archive is stored as ordered objects under `d256/reports_parts/`, with `manifest.json`. The earlier single-object upload failed and must not be used. Download the parts and run `python scripts/release/restore_report_parts.py parts/manifest.json parts reports_source.zip`; this checks every part and the assembled SHA256. The reconstruction script is also saved at `d256/restore_report_parts.py`. Consult `backup_status.json` and final `verification_s3.json` for pending versus verified objects; never equate an upload launch with backup completion.

No disk deletion or HF publication has been performed.


D256 verification complete: all30report parts match S3 SHA256/size. Step8000 optimizer state was downloaded and inspected (712 parameter moments). Final closeout metadata and Git history: `d256/final_closeout.zip`. Both machines/disks may be removed for preserving these training/research assets; no full OS image is retained. Reinstall environments later. No deletion performed.
