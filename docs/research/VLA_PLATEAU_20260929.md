# Why validation loss stopped moving (2026-09-29)

**NO CODE CHANGED.** Research note for the D169b run (lr 5e-5, sim val 0.7372 ->
0.7431 -> 0.7375 -> 0.7365 at updates 1/500/1000/1500; real val 2.621 -> 2.632).

## Causes, ranked by evidence

| # | Cause | Evidence | Cheap test |
|---|---|---|---|
| 1 | **Validation cross-entropy is a poor gauge of flying.** Pilot actions are noisy, so the token loss has a high floor; the most likely action can keep improving while the loss barely moves. | Our own D165: real validation loss 2.565 -> 2.582 (flat) while closed-loop nDTW went 0.128 -> 0.333. OpenVLA stopped on training action-token accuracy (>95%) and robot success, not loss. | Add teacher-forced action-token accuracy to validation; judge checkpoints by closed-loop success on a task subset. |
| 2 | **Far too little training.** | OpenVLA-UAV: LoRA r32, batch 32, lr 5e-4, 200,000 steps (UAV-Flow paper, Table 3) = 6.4 M examples. Ours in total: ~6,000 updates = ~190 k examples (3%). OpenVLA needed 27 epochs; "typical VLM runs of 1-2 epochs" were not enough. | Longer run; no new data needed first. |
| 3 | **Learning rate too low for LoRA, and repeated re-warming.** | LoRA's best LR is ~10x full fine-tuning (Thinking Machines, "LoRA Without Regret"); the official recipe uses 5e-4. 5e-5 is 10x below it. The loss bump after re-warming is the known "stability gap": it recovers and leaves no lasting harm (Gupta et al. 2308.04014; Ibrahim et al. 2403.08763). Stopping the 2e-4 run at update 500 because of that bump was premature (my call). | One long run at the recipe LR with a constant (warmup-stable-decay) schedule: one warmup, no restarts, decay only at the end. |
| 4 | LoRA capacity / model size | LoRA r32 learns less than full fine-tuning as data grows (Biderman et al. 2405.09673), as a slower rate, not a hard floor. OpenVLA-UAV (7B) uses the same r32. | Only if 1-3 are ruled out: rank 64, or compare with a larger base. |

## What this changes

- The early-stop rule on validation loss (D169) is the wrong trigger for this
  problem: it would stop runs that are still improving in flight. Keep loss as a
  guard against divergence, not as the stop signal.
- The user's goal is real flights; real-data training was still improving at the
  end of the 10-shard run (2.94 -> 2.57 over 2,500 updates), so more real shards
  plus a longer run is supported, with the simulator used for testing.

## Sources

- UAV-Flow training configs: 2505.15725, appendix C, Table 3 (OpenVLA-UAV: batch 32, lr 5e-4, LoRA r32, 200k steps)
- OpenVLA (2406.09246): 27 epochs, train until action-token accuracy >95%
- LoRA Without Regret: https://thinkingmachines.ai/blog/lora/
- LoRA Learns Less and Forgets Less: https://arxiv.org/abs/2405.09673
- Continual pre-training, re-warming: https://arxiv.org/abs/2308.04014 , https://arxiv.org/abs/2403.08763
- Warmup-stable-decay schedules: https://arxiv.org/abs/2410.05192
