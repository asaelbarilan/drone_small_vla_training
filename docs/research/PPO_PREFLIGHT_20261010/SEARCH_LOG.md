# Search and inspection log — 2026-10-10

Scope and stopping rule were recorded in PROTOCOL.md before searches. This is a decision-focused review, not a claim of exhaustive systematic coverage. No study was reproduced and no external training package or weights were installed.

## Search lanes

Queries included RL4VLA reinforcement learning vision language action PPO GRPO; RIPT VLA SimpleVLA RL; WorldVLN UAV Flow reinforcement learning; RLinf ConRFT VLA; exact titles for RL4VLA, RIPT-VLA, VLA-RL and RLinf; PPO 1707.06347; Implementation Matters; Time Limits reinforcement learning Pardo; What Matters On Policy; RoboTwin 2 LIBERO; 37 implementation details PPO; Statistical Precipice; Ng reward shaping. Searches used the web search tool followed by primary paper full text and official repositories. Recent-source searches covered 2023–2026; foundational searches were unrestricted.

Backward/cross-reference inspection connected RL4VLA to PPO, GAE and OpenVLA; SimpleVLA-RL to GRPO/DeepSeekMath and RIPT; UAV-Flow to action chunks and benchmark definitions; implementation concerns to truncation and statistical evaluation work. Search-result summaries and abstracts alone were not used for central recommendations.

## Coverage

31 candidates are recorded in candidates.json; 27 sources retained in evidence.json: 25 papers, one author technical article, one official documentation source. Inspection means the relevant methods/results passages and resource sections, not every page of every paper. The readable claim map is EVIDENCE.md; resource/license status is RESOURCES.md. Counts are coverage aids, not quality guarantees.

Primary paper views used arXiv HTML/PDF, PMLR and JMLR. Official repository READMEs/license pages were inspected for RL4VLA, SimpleVLA-RL, RIPT-VLA, ConRFT, WorldVLN, RLinf, UAV-Flow, LIBERO, RoboTwin, rliable, OpenVLA-OFT and OpenVLA. A code license does not clear dataset, simulator, base-model or weight terms.

## Access limitations and exclusions

LIBERO's initially requested version/PDF did not yield usable text; the v1 HTML did. RoboTwin project PDF was replaced by arXiv v2 HTML. Empirical Design HTML was replaced by the JMLR PDF. PPO HTML exposed only a wrapper, so the PDF was inspected. The Ng shaping PDF extracted corrupted text and its screenshot did not provide an inspectable result; that candidate is deferred, not counted as deeply inspected or used to assert a shaping theorem. TRPO, LIBERO-VPro and Batch Online Robotics were discovery/background candidates, not deeply retained evidence. No citation-count ranking was used.

Local inspection covered the server, PPO/GRPO trainer, rollout conversion, reward implementation, collection/evaluation launchers, D209 completion artifacts and D210 audit. Local defects are supported by those files, not inferred from papers. Exact cloud hardware viability remains unmeasured.
