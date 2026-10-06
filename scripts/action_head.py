"""D201: continuous action head (OpenVLA-OFT style) instead of 32 action tokens.

OpenVLA-OFT (2502.19645): bin tokens generated one by one 76.5 % -> all steps in one pass 90.2 %
-> continuous values from a 4-layer MLP with an L1 loss 95.3 %, and about 26x faster. Here, after
the text part of the answer (box lines, progress line), K "slot" tokens are appended; one forward
pass gives their last-layer hidden states, and a small MLP (shared over the K slots) maps each to
one step (dx, dy, dz, dyaw) in the tokenizer's normalised range [-1, 1] (tanh). L1 loss against the
normalised target. The box and progress lines stay text (goal memory / deadband need the line).

Deviation from OFT: the slots attend causally (each sees the slots before it), not
bidirectionally - the Qwen3-VL attention / M-RoPE path is kept unchanged. Still one forward pass.
The slot token is the middle action-bin id: an id the head model never generates as text.
"""

import numpy as np
import torch
from torch import nn

HEAD_FILE = "action_head.pt"


class ActionHead(nn.Module):
    def __init__(self, hidden, channels=4, width=1024):
        super().__init__()
        self.hidden, self.channels, self.width = hidden, channels, width
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, channels),
        )

    def forward(self, slot_hidden):
        """[N, K, hidden] -> [N, K, channels] in [-1, 1]."""
        return torch.tanh(self.net(slot_hidden.float()))

    def save(self, path):
        torch.save(dict(hidden=self.hidden, channels=self.channels, width=self.width, state=self.state_dict()), path)

    @classmethod
    def load(cls, path, device="cpu"):
        blob = torch.load(path, map_location=device, weights_only=False)
        head = cls(blob["hidden"], blob["channels"], blob["width"])
        head.load_state_dict(blob["state"])
        return head.to(device)


def slot_id(tokenizer):
    """The middle action-bin token (an ActionTokenizer)."""
    return tokenizer.vocab_size - tokenizer.n_bins // 2


def bins_of(tokenizer, normalised):
    """Normalised values -> bin index, the same rule as ActionTokenizer.token_ids."""
    return np.digitize(np.asarray(normalised, dtype=float), tokenizer.bins)


def slot_states(model_out, slot_mask, k):
    """Last-layer hidden states at the slot positions -> [rows with slots, k, hidden]."""
    hidden = model_out.hidden_states[-1]
    picked = hidden[slot_mask.bool()]
    return picked.view(-1, k, hidden.shape[-1])
