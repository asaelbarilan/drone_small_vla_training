# D254: training closeout and step-8000 backup

## Final retained model

Retain D207 stage-2 **adapter_s8000**, evaluated as D208: historical validation82/100, nDTW0.4255. These are the existing100-task validation metrics, not the273-task test benchmark; circle endpoint/route limitations documented in D210 remain. This is the user's requested supervised step-8000 checkpoint, not an experimental PPO adapter.

S3 object prefix: `$VLA_BUCKET/d207/stage2/adapter_s8000/` contains adapter_model.safetensors, adapter_config.json and original README.md. Training-resume state is separately present under d207/stage2/checkpoint/state.pt. All31 d207 objects and D240 final PPO checkpoint/evidence objects were listed successfully. The two step-8000 model/config files were downloaded anew from S3 and their SHA256 values matched the retained local checkpoint exactly; see step8000_verification.json. Existing optimizer state presence/size verified, not deserialized in this check.

Verified local S3 download: hf_step8000/. Model weights SHA25612e43d18489531d7446219bc88ed5ede0371935649b1464674a41e1cc155ac8e;322279296bytes. Config SHA2565869411e81804b7a85195b80e9affd8485989f62f566997de92a35303bf03127;1242bytes.

This is a PEFT/LoRA adapter, not standalone full4B base weights. Base model is Qwen/Qwen3-VL-4B-Instruct; local snapshot ebb281ec70b05090aa6165b016eac8ec08e71b17. The original config/model-card contains a machine-local base-model path and the original README is an unfilled template. Preserve these archival originals. Before Hugging Face publication, make a separate portable export with correct base-model metadata, a real model card and the action-tokenization/normalization/inference requirements from this repository. No Hugging Face repository has been created or published by this closeout.

## What did not work

- DAgger D209 completed300updates and100validationtasks. Historical aggregate83/100 versus82/100 was not reliable general improvement. Grounding metrics worsened; orbit references/corrective action cadence exceeded the original move budget, fresh orbit completion/stopping examples were absent, and a phase-alias label defect was demonstrated. Endpoint-only stationary circle successes distorted comparisons. Preserve D209 but do not promote it automatically.
- PPO produced real committed updates and changed actions, but D240's four-update comparison showed mixed behavior, not reliable improvement. Turn reward worsened1.910751->1.843995; ascent reward rose1.453934->1.474038 but maximum route error worsened3.794895->4.392329m and ascent remained incomplete.
- Model-reload drift was found and fixed/verified; the overly restrictive per-sample KL rule was replaced with the agreed full-batch guard after local tests. These were implementation repairs, not evidence of successful policy learning.
- Reward/goal/stopping fixes passed local controls but did not establish a robust learning system. D251 critic failed calibration/stopping/continuation/scale checks. D253's frozen state-aware critic gave positive credit for three observed successful stops, but useful continuation passed only6/8=75% against90% required. Five of six evaluation flights were saved, two were timeouts, no completed premature stop was available; final early-stop coverage remained unvalidated. Independently verified partial evidence, no-go for AWS. No further patch/retry is scheduled.

## Paid machines and continuing charges

EC2 independently confirms both instances already STOPPED: Windows i-03a3c8632314bf5fc and Linux training i-05f21d6200f5c1045. No start, termination, volume deletion or new training was performed. No Elastic IPs in the checked region. User explicitly requested retaining disks.

Preserved gp3 volumes: Windows250GiB and training150GiB, both3000IOPS/125MiBps baseline. AWS Price List effective2026-10-01:USD0.08/GB-month inus-east-1. Calculated storage estimate:USD20+USD12=USD32/month combined, orUSD1.07/day using30days. This excludes S3/snapshots/taxes/commitments/other resources and is not a full-account bill. Stopped compute is not billed as running compute; retained EBS storage remains billed. Terminating an instance while retaining its disk does not eliminate this storage cost. No deletion authorized.

## Archive

All existing reports plus docs/scripts/tests and root project documentation are being archived to d254/closeout_20261015/reports_source.zip with a per-file SHA256 inventory. The archive excludes its own output, transient Python bytecode and the separately verified duplicate hf_step8000 download. Backup verification is recorded separately after upload. No environment/key files are included. Existing S3 step8000 and PPO checkpoints remain untouched.
