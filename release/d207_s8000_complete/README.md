---
base_model: Qwen/Qwen3-VL-4B-Instruct
library_name: transformers
tags:
- robotics
- vision-language-action
- drone
---
# Complete drone VLA: D207 step8000

This is a complete, single-download VLA package: the original Qwen3-VL-4B-Instruct base weights, trained step8000 LoRA weights, processor/tokenizer, action normalization, and inference code are included. No separate base-model download is required. Base and adapter are intentionally stored separately inside the same package; they are loaded together at inference. This avoids introducing an unverified merged/quantized export. It is not a standard text-generation pipeline or a standalone AutoModel action API.

## Run

Install `requirements.txt` in a CUDA-capable Python environment, then:

```sh
python serve.py --precision bf16
```

BF16 is the evaluated configuration; NF4 is an alternative with different numerical behavior, not a claim of exact equivalence. CPU inference is not supported by this server. The package has not yet passed a new end-to-end GPU release smoke test. No cloud training or flight is launched by packaging.

The server exposes POST `/reset` and POST `/predict`. Predict accepts `image` (base64 PNG), `proprio` ([x,y,z,yaw_degrees] in the simulator start frame, position in centimetres), and `instr`. It returns eight start-frame poses with yaw in radians. Internally the model predicts drone-frame dx,dy,dz,dyaw tokens; the included server handles normalization, integration, box/progress prompts, goal memory and arrival deadband. A real drone needs its own tested control/safety interface. The Unreal simulator is not part of this model package.

## Provenance and results

Base revision: ebb281ec70b05090aa6165b016eac8ec08e71b17. Trained adapter SHA256: 12e43d18489531d7446219bc88ed5ede0371935649b1464674a41e1cc155ac8e. D207 supervised stage2 checkpoint8000 was evaluated as D208: historical100-task development validation82/100 andnDTW0.4255. These are not a new273-task benchmark result; endpoint-only circle scoring has known limitations. The package uses the supervised model, not the later experimental PPO policy. PPO/critic experiments did not establish reliable improvement.

`package_manifest.json` records hashes for every included file. Optimizer state and training data are kept in the private training backup, not required for inference and not included in this public-release candidate.

## Publication status and licensing

Prepared locally for later Hugging Face publication; not yet published. Qwen's base model card declares Apache-2.0. Preserve upstream license/notices before publication. UAV-Flow data licensing and the final derivative-model license require review; this draft does not invent a license grant. Full release smoke verification and license packaging remain publication checks, not training gates.
