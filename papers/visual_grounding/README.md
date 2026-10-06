# Visual grounding / visual under-reliance in VLAs (D192 reading list)

PDFs are local only (`*.pdf` is gitignored); links in `docs/research/VLA_VISUAL_GROUNDING_20261004.md`.

- `2024_ECoT_embodied-chain-of-thought.pdf` - ECoT (Zawalski et al., CoRL 2024): the VLA writes plan, sub-task, **object bounding boxes** and gripper position before its action; +28 points absolute on OpenVLA generalization tasks. Took: a grounded box before the action makes the policy "look".
- `2025_ECoT-Lite_efficient-embodied-reasoning.pdf` - ECoT-Lite (Chen et al., CoRL 2025): reasoning helps mainly through better representations; train with reasoning, **drop it at test time** (reasoning dropout) and keep most of the gain at 3x speed. Took: the box can be a training-only target.
- `2026_LIT_breaking-vision-action-shortcut.pdf` - LIT (2609.12641): an image-free action prior conditioned on the terminal goal pose, with vision entering only through tokens supervised to predict that pose. Took: our progress line is already this kind of goal bottleneck; the weak part is predicting it from the photo.
- `2026_ReViP_vision-proprioception-rebalance.pdf` - ReViP (2601.16667): policies over-trust state and ignore visible failure; rebalances vision vs state (+26 % over pi0). Not our main failure (Turn's first call has zero state).
- `2026_StateVLA_how-to-use-proprioceptive-state.pdf` - 2608.03052: five state interfaces compared; state as a text prompt is fine for a single frame (57.7 % vs 54.6 % without). Not a cause here.
- `2026_CosFly-VLA_spatially-aware-uav-tracking.pdf` - CosFly-VLA (2607.15004): UAV tracking VLA on a Qwen3.5 backbone with LoRA and a separate target-box head next to the action head. Took: drone VLAs on Qwen add explicit target grounding.
- `2025_OBEYED-VLA_object-centric-grounding.pdf` - OBEYED-VLA (2512.22519): a VLM grounding stage selects the task object before the VLA acts; robust to clutter and absent targets. Took: object-centric grounding as a separate stage (heavier option).
- `2026_ST4VLA_spatially-guided-training.pdf` - ST4VLA (2602.10109, Qwen2.5-VL): action-only training destroys grounding, plain co-training only partly keeps it; staged grounding pre-training -> spatially guided action post-training (+18 pts). Took: the staged order and damping the action gradient into the grounding path.
- `2026_GroundingSFT_why-grounding-hurts-medical-vqa.pdf` - 2604.27720: answer-only SFT removes box output; mixed grounding supervision restores it. Took: our D192 observation is a known effect.
- `2026_BeTTER_illusion-of-embodied-reasoning.pdf` - 2604.18000: VLA reasoning does not drive actions under interventions (shortcuts). Took: test line-vs-box consistency.
- `2026_VLAfaithfulness_chain-of-causation-driving.pdf` - 2605.17268: reasoning-action consistency 53 % in a driving VLA. Took: measure consistency explicitly.
