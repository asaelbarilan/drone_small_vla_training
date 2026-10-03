"""Qwen3-VL-4B loading for the VLA trainer and server, without the architecture testbed.

Same code as run_qwen_frd_overfit.MODEL / load_base / cuda (the helpers the trainer used to
borrow); the trainer's bf16 path loads the base itself and only needs MODEL, the two
transformers classes and cuda(). QWEN_MODEL points at a local snapshot of
Qwen/Qwen3-VL-4B-Instruct.
"""

import os
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration  # noqa: F401

MODEL = Path(
    os.environ.get(
        "QWEN_MODEL",
        "D:/drone_vla_pilot/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17",
    )
)


def load_base():
    """NF4 base (the 8 GB laptop path); norm weights upcast to fp32 as before."""
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(MODEL),
        local_files_only=True,
        quantization_config=config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    for name, parameter in model.named_parameters():
        if "norm" in name and parameter.dtype == torch.bfloat16:
            parameter.data = parameter.data.float()
    return model


def cuda(batch):
    return {k: v.to("cuda") for k, v in batch.items()}
