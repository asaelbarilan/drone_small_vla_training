# Action heads instead of action tokens (D200 reading list)

PDFs are local only (`*.pdf` is gitignored); notes in `docs/research/VLA_ACTION_HEAD_20261006.md`.

- `2025_OpenVLA-OFT_fine-tuning-speed-and-success.pdf` - OpenVLA-OFT (2502.19645): LIBERO
  76.5 % (autoregressive bin tokens) -> 90.2 % (+ parallel decoding of a chunk) -> 95.3 %
  (+ continuous actions, L1 loss). About 26x faster. 4-layer MLP on the last hidden states of
  empty action slots, LoRA r32. Took: the cheapest head design, and the ablation order.
- `2026_VLANeXt_systematic-vla-recipes.pdf` - VLANeXt (2602.18532): from bin tokens (19.8) to a
  policy head with query tokens (64.4), chunking (74.6) and flow matching (80). Took: the size of
  the gain from leaving bin tokens.
- `2026_Qwen-VLA_unified-vla-dit-decoder.pdf` - Qwen-VLA (2605.30280): the official Qwen VLA,
  Qwen3.5-4B plus a 1.15B DiT flow-matching action decoder. Took: Qwen's own team also uses a
  separate continuous decoder; 1.15B is too big for our 8 GB goal.
- `2026_FLIGHT_pilot-like-uav-navigation.pdf` - FLIGHT (2606.06836): UAV VLA on Qwen2.5-VL-3B
  with LoRA and a DiT-B action head (59 % vs OpenVLA-UAV 10.5 % on long-horizon). Took: drone
  VLAs on small Qwen models already use a separate head.
